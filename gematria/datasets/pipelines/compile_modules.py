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

from gematria.datasets.pipelines import compile_modules_lib

_PARQUET_FOLDER = flags.DEFINE_string(
    'parquet_folder',
    None,
    'The path to the folder containing parquet files',
    required=True,
)

_OUTPUT_TXT_FILE = flags.DEFINE_string(
    'output_txt_file', None, 'The path to the output txt file', required=True
)

_REMOVE_MEMORY_ACCESSING_INSTRUCTIONS = flags.DEFINE_bool(
    'remove_memory_accessing_instructions',
    False,
    'Whether to remove memory accessing instructions from the basic blocks.',
)


def main(argv) -> None:
  del argv  # Unused.

  beam_options = pipeline_options.PipelineOptions()

  pipeline_constructor = compile_modules_lib.get_bbs(
      os.path.join(_PARQUET_FOLDER.value, '*.parquet'),
      _OUTPUT_TXT_FILE.value,
      _REMOVE_MEMORY_ACCESSING_INSTRUCTIONS.value,
  )

  with beam.Pipeline(options=beam_options) as pipeline:
    pipeline_constructor(pipeline)


if __name__ == '__main__':
  app.run(main)
