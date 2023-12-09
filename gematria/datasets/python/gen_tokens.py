from gematria.basic_block.python import basic_block
from gematria.basic_block.python import basic_block_protos
from gematria.proto import basic_block_pb2
from gematria.proto import throughput_pb2
from gematria.proto import canonicalized_instruction_pb2
from gematria.io.python import tfrecord

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging

_CanonicalizedInstructionProto = (
    canonicalized_instruction_pb2.CanonicalizedInstructionProto
)

r"""Generates tokens from a Gematria data set.


Usage:
  gen_tokens \
      --gematria_input_tfrecord=/tmp/bhive/skl.tfrecord \
      --gematria_output_tokens=/tmp/bhive/skl_tokens.txt \

"""

_INPUT_TFRECORD_FILE = flags.DEFINE_string(
    'gematria_input_tfrecord',
    None,
    'The name of the TFRecord file to read the tokens from.',
    required=True,
)

_OUTPUT_TOKENS_FILE = flags.DEFINE_string(
    'gematria_output_tokens',
    None,
    'The name of the file to write the tokens to.',
    required=True,
)

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  output_blocks = list(
    tfrecord.read_protos((_INPUT_TFRECORD_FILE.value,), throughput_pb2.BasicBlockWithThroughputProto)
  )
  token_set = set()
  for block in output_blocks:
    print(block)
    for instruction in block.basic_block.canonicalized_instructions:
     ginstruction = basic_block_protos.instruction_from_proto(instruction)
     for token in ginstruction.as_token_list():
       if not token.startswith('%'):
        token_set.add(token)
  print(token_set)
  with open(_OUTPUT_TOKENS_FILE.value, 'w') as f:
    for token in token_set:
      f.write(token)
      f.write('\n')
      
  

if __name__ == '__main__':
  app.run(main)