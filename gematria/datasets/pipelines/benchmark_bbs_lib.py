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

from collections.abc import Callable, Iterable
from typing import Tuple

import apache_beam as beam
from pybind11_abseil import status

from gematria.proto import execution_annotation_pb2
from gematria.datasets.python import exegesis_benchmark


class BenchmarkBasicBlock(beam.DoFn):
  """A Beam function that benchmarks basic blocks."""

  def setup(self):
    self._exegesis_benchmark = exegesis_benchmark.ExegesisBenchmark.create()

  def process(
      self,
      block_with_annotations: execution_annotation_pb2.BlockWithExecutionAnnotations,
  ) -> Iterable[Tuple[str, float]]:
    try:
      benchmark_code = self._exegesis_benchmark.process_annotated_block(
          block_with_annotations
      )
      benchmark_value = self._exegesis_benchmark.benchmark_basic_block(
          benchmark_code
      )
      yield (block_with_annotations.block_hex, benchmark_value)
    except status.StatusNotOk:
      pass


class FormatBBsForOutput(beam.DoFn):
  """A Beam function for formatting hex/throughput values for output."""

  def process(
      self, block_hex_and_throughput: Tuple[str, float]
  ) -> Iterable[str]:
    block_hex, throughput = block_hex_and_throughput
    yield f'{block_hex},{throughput}'


def benchmark_bbs(
    input_file_pattern: str, output_file_pattern: str
) -> Callable[[beam.Pipeline], None]:
  """Creates a pipeline to benchmark BBs."""

  def pipeline(root: beam.Pipeline) -> None:
    annotated_bbs = root | 'Load annotated blocks' >> beam.io.ReadFromTFRecord(
        input_file_pattern,
        coder=beam.coders.ProtoCoder(
            execution_annotation_pb2.BlockWithExecutionAnnotations().__class__
        ),
    )
    annotated_bbs_shuffled = annotated_bbs | 'Shuffle' >> beam.Reshuffle()
    benchmarked_blocks = annotated_bbs_shuffled | 'Benchmarking' >> beam.ParDo(
        BenchmarkBasicBlock()
    )
    formatted_output = benchmarked_blocks | 'Formatting' >> beam.ParDo(
        FormatBBsForOutput()
    )

    _ = formatted_output | 'Write To Text' >> beam.io.WriteToText(
        output_file_pattern
    )

  return pipeline
