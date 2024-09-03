# Copyright 2024 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable, Iterable, Sequence
import os
import subprocess

from absl import logging
import apache_beam as beam
from rules_python.python.runfiles import runfiles

from gematria.datasets.python import extract_bbs_from_obj


def _get_llvm_binary_path(tool_name: str) -> str:
  runfiles_env = runfiles.Create(os.environ)
  assert runfiles_env is not None
  return runfiles_env.Rlocation(os.path.join('llvm-project/llvm', tool_name))


class OptimizeModules(beam.DoFn):
  """A Beam function that uses LLVM opt to optimize bitcode modules."""

  def __init__(self, optimization_pipelines: Sequence[str]):
    self._optimization_pipelines = optimization_pipelines
    self._opt_path = _get_llvm_binary_path('opt')

  def optimize_module(
      self, input_module: bytes, optimization_pipeline: str
  ) -> bytes:
    command_vector = [self._opt_path, f'-passes={optimization_pipeline}']
    result = subprocess.run(
        command_vector, input=input_module, capture_output=True, check=True
    )
    return result.stdout

  def process(self, input_module: bytes) -> Iterable[bytes]:
    for optimization_pipeline in self._optimization_pipelines:
      try:
        yield self.optimize_module(input_module, optimization_pipeline)
      except subprocess.CalledProcessError as process_error:
        logging.error(process_error)
        continue


class LowerModulesAsm(beam.DoFn):
  """A Beam function that lowers bitcode files to object files."""

  def __init__(self, optimization_levels: Sequence[str]):
    self._optimization_levels = optimization_levels
    self._llc_path = _get_llvm_binary_path('llc')

  def lower_module(self, optimization_level: str, input_module: bytes) -> bytes:
    command_vector = [
        self._llc_path,
        optimization_level,
        '-filetype=obj',
        '-basic-block-sections=labels',
    ]
    result = subprocess.run(
        command_vector, input=input_module, capture_output=True, check=True
    )
    return result.stdout

  def process(self, input_module: bytes) -> Iterable[bytes]:
    for optimization_level in self._optimization_levels:
      try:
        yield self.lower_module(optimization_level, input_module)
      except subprocess.CalledProcessError as process_error:
        logging.error(process_error)
        continue


class GetBBsFromModule(beam.DoFn):
  """A Beam function to extract BB hex values from object files."""

  def process(self, input_object_file: bytes) -> Iterable[str]:
    for bb_hex_value in extract_bbs_from_obj.get_basic_block_hex_values(
        input_object_file
    ):
      yield bb_hex_value


class DeduplicateBBs(beam.ptransform.PTransform):
  """A Beam transform to deduplicate string data."""

  def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
    return (
        pcoll
        | 'Use Value as Key'
        >> beam.Map(lambda bb_hex_value: (bb_hex_value, None))
        | 'Deduplicate' >> beam.CombinePerKey(lambda values: next(iter(values)))
        | 'Output Value'
        >> beam.Map(lambda bb_hex_value_tuple: bb_hex_value_tuple[0])
    )


def get_bbs(
    input_file_pattern: str, output_file: str
) -> Callable[[beam.Pipeline], None]:
  """Creates a pipeline to process BBs from IR modules.

  This function returns a function that builds a beam pipeline to automatically
  load ir files from a ComPile style parquet file, process them into assembly
  basic blocks, deduplicate them, and then write them to a text file.

  Args:
    input_file_pattern: A grep-like pattern to use to search for the parquet
      files to process.
    output_file: The output file pattern to use when writing the basic blocks
      to disk.

  Returns:
    A function that accepts a beam pipeline and adds on all the steps needed
    to process the input IR modules.
  """

  def pipeline(root: beam.Pipeline) -> None:
    parquet_data = root | 'Read' >> beam.io.ReadFromParquet(
        input_file_pattern, columns=['content']
    )
    module_data = parquet_data | 'Load' >> beam.Map(
        lambda parquet_row: parquet_row['content']
    )
    module_data_shuffled = module_data | 'Shuffle' >> beam.Reshuffle()
    optimized_modules = module_data_shuffled | 'Optimize' >> beam.ParDo(
        OptimizeModules(
            ['default<O0>', 'default<O1>', 'default<O2>', 'default<O3>']
        )
    )
    lowered_modules = optimized_modules | 'Lower' >> beam.ParDo(
        LowerModulesAsm(['-O0', '-O1', '-O2', '-O3'])
    )
    bb_hex_values = lowered_modules | 'Get BBs' >> beam.ParDo(
        GetBBsFromModule()
    )
    bb_hex_values_deduplicated = (
        bb_hex_values | 'Deduplicate' >> DeduplicateBBs()
    )

    _ = bb_hex_values_deduplicated | 'WriteToText' >> beam.io.WriteToText(
        output_file
    )

  return pipeline
