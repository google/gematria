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

import os

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.options import pipeline_options

from gematria.datasets.pipelines import benchmark_bbs_lib
from gematria.datasets.pipelines import benchmark_cpu_scheduler

_INPUT_FILE_PATTERN = flags.DEFINE_string(
    'input_file_pattern',
    None,
    'The input file pattern to load annotated blocks from.',
    required=True,
)
_OUTPUT_FILE_PATTERN = flags.DEFINE_string(
    'output_file_pattern', None, 'The output file path/pattern.', required=True
)
_BENCHMARK_SCHEDULER = flags.DEFINE_enum_class(
    'benchmark_scheduler',
    benchmark_cpu_scheduler.BenchmarkSchedulerImplementations.NO_SCHEDULING,
    benchmark_cpu_scheduler.BenchmarkSchedulerImplementations,
    'The scheduler to use for choosing a core for running benchmarks.',
)


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  beam_options = pipeline_options.PipelineOptions()

  pipeline_constructor = benchmark_bbs_lib.benchmark_bbs(
      _INPUT_FILE_PATTERN.value,
      _OUTPUT_FILE_PATTERN.value,
      _BENCHMARK_SCHEDULER.value,
  )

  with beam.Pipeline(options=beam_options) as pipeline:
    pipeline_constructor(pipeline)


if __name__ == '__main__':
  app.run(main)
