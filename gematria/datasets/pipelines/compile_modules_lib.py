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
from apache_beam import metrics
from pybind11_abseil import status

from gematria.datasets.python import extract_bbs_from_obj
from gematria.datasets.python import process_and_filter_bbs
from gematria.datasets.python import bhive_importer
from gematria.datasets.python import bhive_to_exegesis
from gematria.llvm.python import canonicalizer
from gematria.llvm.python import llvm_architecture_support
from gematria.proto import execution_annotation_pb2
from gematria.basic_block.python import basic_block_protos
from gematria.basic_block.python import basic_block
import gematria.llvm.python.runfiles

_BEAM_METRIC_NAMESPACE_NAME = 'compile_modules'


class OptimizeModules(beam.DoFn):
  """A Beam function that uses LLVM opt to optimize bitcode modules."""

  def __init__(self, optimization_pipelines: Sequence[str]):
    self._optimization_pipelines = optimization_pipelines
    self._modules_succeeded = metrics.Metrics.counter(
        _BEAM_METRIC_NAMESPACE_NAME, 'optimize_modules_success'
    )
    self._modules_failed = metrics.Metrics.counter(
        _BEAM_METRIC_NAMESPACE_NAME, 'optimize_modules_failure'
    )

  def setup(self):
    self._opt_path = gematria.llvm.python.runfiles.get_llvm_binary_path('opt')

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
        self._modules_succeeded.inc()
      except subprocess.CalledProcessError as process_error:
        logging.error(process_error)
        self._modules_failed.inc()
        continue


class LowerModulesAsm(beam.DoFn):
  """A Beam function that lowers bitcode files to object files."""

  def __init__(self, optimization_levels: Sequence[str]):
    self._optimization_levels = optimization_levels
    self._modules_succeded = metrics.Metrics.counter(
        _BEAM_METRIC_NAMESPACE_NAME, 'lower_modules_success'
    )
    self._modules_failed = metrics.Metrics.counter(
        _BEAM_METRIC_NAMESPACE_NAME, 'lower_modules_failure'
    )

  def setup(self):
    self._llc_path = gematria.llvm.python.runfiles.get_llvm_binary_path('llc')

  def lower_module(self, optimization_level: str, input_module: bytes) -> bytes:
    command_vector = [
        self._llc_path,
        optimization_level,
        '-filetype=obj',
        '-basic-block-address-map',
    ]
    result = subprocess.run(
        command_vector, input=input_module, capture_output=True, check=True
    )
    return result.stdout

  def process(self, input_module: bytes) -> Iterable[bytes]:
    for optimization_level in self._optimization_levels:
      try:
        yield self.lower_module(optimization_level, input_module)
        self._modules_succeded.inc()
      except subprocess.CalledProcessError as process_error:
        logging.error(process_error)
        self._modules_failed.inc()
        continue


class GetBBsFromModule(beam.DoFn):
  """A Beam function to extract BB hex values from object files."""

  def __init__(self):
    self._bbs_produced = metrics.Metrics.counter(
        _BEAM_METRIC_NAMESPACE_NAME, 'extracted_bbs'
    )

  def process(self, input_object_file: bytes) -> Iterable[str]:
    for bb_hex_value in extract_bbs_from_obj.get_basic_block_hex_values(
        input_object_file
    ):
      yield bb_hex_value
      self._bbs_produced.inc()


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


class ProcessAndFilterBBs(beam.DoFn):
  """A Beam transform to process and filter BBs."""

  def __init__(self, remove_memory_accessing_instructions: bool):
    self._remove_memory_accessing_instructions = (
        remove_memory_accessing_instructions
    )
    self._blocks_after_filtering = metrics.Metrics.counter(
        _BEAM_METRIC_NAMESPACE_NAME, 'filtered_bbs'
    )

  def setup(self):
    self._bb_processor_filter = process_and_filter_bbs.BBProcessorFilter()

  def process(self, bb_hex: str) -> Iterable[str]:
    output_block = self._bb_processor_filter.remove_risky_instructions(
        bb_hex, bb_hex, self._remove_memory_accessing_instructions
    )
    if output_block != '':
      yield output_block
      self._blocks_after_filtering.inc()


class AnnotateBBs(beam.DoFn):

  def __init__(
      self,
      annotator_type: bhive_to_exegesis.AnnotatorType,
      max_annotation_attempts: int,
  ):
    self._annotator_type = annotator_type
    self._max_annotation_attempts = max_annotation_attempts
    self._blocks_annotated_successfully = metrics.Metrics.counter(
        _BEAM_METRIC_NAMESPACE_NAME, 'annotate_blocks_success'
    )
    self._blocks_failed_annotation = metrics.Metrics.counter(
        _BEAM_METRIC_NAMESPACE_NAME, 'annotate_blocks_failure'
    )

  def setup(self):
    self._x86_llvm = llvm_architecture_support.LlvmArchitectureSupport.x86_64()
    self._bhive_to_exegesis = bhive_to_exegesis.BHiveToExegesis.create(
        self._x86_llvm
    )

  def process(
      self, bb_hex: str
  ) -> Iterable[execution_annotation_pb2.BlockWithExecutionAnnotations]:
    # Do a dummy write. Otherwise some interaction with beam prevents us from
    # writing to file descriptors within the annotator subprocess.
    # TODO(boomanaiden154): This should be investigated and fixed properly.
    with open('/dev/null', 'w') as dummy_file:
      print('', file=dummy_file)

    try:
      yield execution_annotation_pb2.BlockWithExecutionAnnotations(
          execution_annotations=self._bhive_to_exegesis.annotate_basic_block(
              bb_hex, self._annotator_type, self._max_annotation_attempts
          ),
          block_hex=bb_hex,
      )
      self._blocks_annotated_successfully.inc()
    except status.StatusNotOk:
      self._blocks_failed_annotation.inc()
      pass


class GetVocab(beam.DoFn):
  """A Beam transform to get vocab from basic blocks."""

  def setup(self):
    self._x86_llvm = llvm_architecture_support.LlvmArchitectureSupport.x86_64()
    self._x86_canonicalizer = canonicalizer.Canonicalizer.x86_64(self._x86_llvm)
    self._importer = bhive_importer.BHiveImporter(self._x86_canonicalizer)

  def process(self, bb_hex: str) -> Iterable[str]:
    raw_block_proto = self._importer.basic_block_proto_from_hex(bb_hex)
    block_proto = basic_block_protos.basic_block_from_proto(raw_block_proto)
    tokens = set()

    for instruction in block_proto.instructions:
      tokens.update(instruction.as_token_list())

    yield from tokens


def get_bbs(
    input_file_pattern: str,
    output_file: str,
    remove_memory_accessing_instructions: bool,
    annotator_type: bhive_to_exegesis.AnnotatorType,
    max_annotation_attempts: int,
) -> Callable[[beam.Pipeline], None]:
  """Creates a pipeline to process BBs from IR modules.

  This function returns a function that builds a beam pipeline to automatically
  load IR files from a ComPile style Parquet file, process them into assembly
  basic blocks, deduplicate them, and then write them to a text file.

  Args:
    input_file_pattern: A grep-like pattern to use to search for the Parquet
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
    processed_filtered_bbs = (
        bb_hex_values_deduplicated
        | 'Filter'
        >> beam.ParDo(ProcessAndFilterBBs(remove_memory_accessing_instructions))
    )
    processed_bbs_deduplicated = (
        processed_filtered_bbs | 'Deduplicate Processed BBs' >> DeduplicateBBs()
    )
    annotated_bbs = processed_bbs_deduplicated | 'Annotate BBs' >> beam.ParDo(
        AnnotateBBs(annotator_type, max_annotation_attempts)
    )

    _ = annotated_bbs | 'Write annotated blocks' >> beam.io.WriteToTFRecord(
        output_file,
        coder=beam.coders.ProtoCoder(
            execution_annotation_pb2.BlockWithExecutionAnnotations().__class__
        ),
    )

  return pipeline
