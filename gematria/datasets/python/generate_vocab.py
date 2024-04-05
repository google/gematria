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

r"""Creates a token list from a Gematria dataset.

Reads basic blocks from a TFRecord, gets the tokens for each basic block,
and produces a text file containing a new-line separated list of all
the unique tokens that compose the dataset vocab.

Usage:
  generate_vocab \
      --input_tfrecord=/tmp/dataset.tfrecord \
      --output_vocab=/tmp/vocab.txt
"""

from absl import app
from absl import flags
from absl import logging

from gematria.io.python import tfrecord
from gematria.proto import throughput_pb2
from gematria.basic_block.python import basic_block_protos
from gematria.basic_block.python import basic_block

_INPUT_TFRECORD_FILE = flags.DEFINE_string(
    'input_tfrecord',
    None,
    'The path to the tfrecord file to process',
    required=True,
)
_OUTPUT_TXT_FILE = flags.DEFINE_string(
    'output_vocab', None, 'The path to the output txt file', required=True
)


def main(_) -> None:
  loaded_protos = tfrecord.read_protos(
      _INPUT_TFRECORD_FILE.value, throughput_pb2.BasicBlockWithThroughputProto
  )

  tokens = set()

  current_proto_index = 0

  for proto in loaded_protos:
    if current_proto_index % 1000 == 0:
      logging.info(f'Just finished proto {current_proto_index}')
    basic_block_proto = basic_block_protos.basic_block_from_proto(
        proto.basic_block
    )
    for instruction in basic_block_proto.instructions:
      tokens.update(instruction.as_token_list())

    current_proto_index += 1

  with open(_OUTPUT_TXT_FILE.value, 'w') as output_file_handle:
    for token in tokens:
      output_file_handle.write(f'{token}\n')


if __name__ == '__main__':
  app.run(main)
