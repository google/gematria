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

import apache_beam as beam
from rules_python.python.runfiles import runfiles
import os

from absl import logging

from collections.abc import Iterable
from collections.abc import Callable

import subprocess

from gematria.datasets.python import extract_bbs_from_obj


def _get_llvm_binary_path(tool_name: str) -> str:
  runfiles_env = runfiles.Create(os.environ)
  assert runfiles_env is not None
  return runfiles_env.Rlocation('llvm-project/llvm/' + tool_name)


class OptimizeModules(beam.DoFn):

  def __init__(self, optimization_pass_lists: list[str]):
    self.optimization_pass_lists = optimization_pass_lists
    self.opt_path = _get_llvm_binary_path('opt')

  def optimize_module(
      self, input_module: bytes, optimization_pass_list: list[str]
  ) -> bytes:
    command_vector = [self.opt_path, f'-passes={optimization_pass_list}']
    with subprocess.Popen(
        command_vector,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as opt_process:
      (output_bc, stderr) = opt_process.communicate(input=input_module)
      if opt_process.returncode != 0:
        logging.error(stderr)
        raise ValueError('Expected opt to return 0')
      return output_bc

  def process(self, input_module: bytes) -> Iterable[bytes]:
    for optimization_pass_list in self.optimization_pass_lists:
      try:
        yield self.optimize_module(input_module, optimization_pass_list)
      except ValueError:
        continue


class LowerModulesAsm(beam.DoFn):

  def __init__(self, optimization_levels: list[str]):
    self.optimization_levels = optimization_levels
    self.llc_path = _get_llvm_binary_path('llc')

  def lower_module(self, optimization_level: str, input_module: bytes) -> bytes:
    command_vector = [
        self.llc_path,
        optimization_level,
        '-filetype=obj',
        '-basic-block-sections=labels',
    ]
    with subprocess.Popen(
        command_vector,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as llc_process:
      (output_obj, stderr) = llc_process.communicate(input=input_module)
      if llc_process.returncode != 0:
        logging.error(stderr)
        raise ValueError('Expected llc to return 0')
      return output_obj

  def process(self, input_module: bytes) -> Iterable[bytes]:
    for optimization_level in self.optimization_levels:
      try:
        yield self.lower_module(optimization_level, input_module)
      except ValueError:
        continue


class GetBBsFromModule(beam.DoFn):

  def process(self, input_object_file: bytes) -> Iterable[str]:
    for bb_hex_value in extract_bbs_from_obj.get_basic_block_hex_values(
        input_object_file
    ):
      yield bb_hex_value


class DeduplicateBBs(beam.ptransform.PTransform):

  def expand(self, pcoll):
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
  def pipeline(root: beam.Pipeline) -> None:
    # Do something to process parquet files here.
    # TODO(boomanaiden154): Only load IR here
    parquet_data = root | 'Read' >> beam.io.ReadFromParquet(input_file_pattern)
    module_data = parquet_data | 'Load' >> beam.Map(
        lambda parquet_row: parquet_row['content']
    )
    optimized_modules = module_data | 'Optimize' >> beam.ParDo(
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
