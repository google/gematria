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

from absl import app
from absl import flags
from absl import logging

import os

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options import pipeline_options

from gematria.datasets.beam import compile_modules_lib

_PARQUET_FOLDER = flags.DEFINE_string(
    'parquet_folder',
    None,
    'The path to the folder containing parquet files',
    required=True,
)

_OUTPUT_TXT_FILE = flags.DEFINE_string(
    'output_txt_file', None, 'The path to the output txt file', required=True
)


def main(argv) -> None:
  del argv

  beam_options = PipelineOptions()

  # beam_options.view_as(pipeline_options.DirectOptions).direct_num_workers = 0
  # beam_options.view_as(pipeline_options.DirectOptions).direct_running_mode = 'in_memory'

  pipeline_constructor = compile_modules_lib.get_bbs(
      _PARQUET_FOLDER.value, _OUTPUT_TXT_FILE.value
  )

  with beam.Pipeline(options=beam_options) as pipeline:
    pipeline_constructor(pipeline)


if __name__ == '__main__':
  app.run(main)
