# Copyright 2023 Google Inc.
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
"""Runs the hierarchical LSTM model.

See the docstring of HierarchicalLstmModel in
gematria/sequence/python/sequence_model_hlstm.py
for more details.
"""

from collections.abc import Sequence

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf

from gematria.basic_block.python import tokens
from gematria.model.python import main_function
from gematria.model.python import oov_token_behavior
from gematria.model.python import token_model_flags
from gematria.sequence.python import sequence_model_hlstm
from gematria.utils.python import flag_utils

_OutOfVocabularyTokenBehavior = oov_token_behavior.OutOfVocabularyTokenBehavior

_TOKEN_EMBEDDING_SIZE = flags.DEFINE_integer(
    'gematria_token_embedding_size',
    256,
    ('The size of the token embedding vectors used in the hierarchical LSTM'
     ' model.'),
)
_INSTRUCTION_EMBEDDING_SIZE = flags.DEFINE_integer(
    'gematria_instruction_embedding_size',
    256,
    ('The size of the instruction embedding vectors used in the hierarchical'
     ' LSTM model.'),
)
_BLOCK_EMBEDDING_SIZE = flags.DEFINE_integer(
    'gematria_block_embedding_size',
    256,
    ('The size of the basic block embedding vectors used in the hierarchical'
     ' LSTM model.'),
)

_OUTPUT_LAYERS = flags.DEFINE_string(
    'gematria_output_layers',
    '',
    ('Specifies a comma-separated string that represents the number of'
     ' shared dense layers and their sizes used to compute the output of the'
     ' network. These dense layers consume the outputs of the'
     ' basic-block-level LSTM. When empty, the outputs of the'
     ' basic-block-level LSTM are passed directly to the input of the'
     ' task-specific dense layers.'),
)
_TASK_OUTPUT_LAYERS = flags.DEFINE_string(
    'gematria_task_output_layers',
    '',
    ('Specifies a comma-separated string that represents the number of '
     ' task-specific dense layers and their sizes used to compute the output'
     ' of the network. The task-specific dense layers consume the output of'
     ' the shared dense layers; they are followed by a single task-specific'
     ' linear layer that produces the output of the network for the task.'
     ' When empty, the linear layer is added directly after the shared dense'
     ' layers or after the basic-block-level LSTM layer.'),
)

_BIDIRECTIONAL = flags.DEFINE_bool(
    'gematria_bidirectional',
    False,
    'Run the basic block level LSTM network in bidirectional mode.',
)

_SEQ_2_SEQ = flags.DEFINE_bool(
    'gematria_seq2seq',
    False,
    ('Run the model in the seq2seq mode where predictions are made for each'
     ' instruction in the basic block rather than for the basic block as a'
     ' whole.'),
)

flags.register_validator(
    _TOKEN_EMBEDDING_SIZE.name,
    flag_utils.is_positive,
    flag_utils.MUST_BE_POSITIVE_ERROR,
)
flags.register_validator(
    _INSTRUCTION_EMBEDDING_SIZE.name,
    flag_utils.is_positive,
    flag_utils.MUST_BE_POSITIVE_ERROR,
)
flags.register_validator(
    _BLOCK_EMBEDDING_SIZE.name,
    flag_utils.is_positive,
    flag_utils.MUST_BE_POSITIVE_ERROR,
)

flags.register_validator(
    _OUTPUT_LAYERS.name,
    flag_utils.is_positive_integer_list,
    flag_utils.MUST_BE_POSITIVE_INTEGER_LIST_ERROR,
)
flags.register_validator(
    _TASK_OUTPUT_LAYERS.name,
    flag_utils.is_positive_integer_list,
    flag_utils.MUST_BE_POSITIVE_INTEGER_LIST_ERROR,
)


def main(argv: Sequence[str]) -> None:
  del argv  # Unused
  out_of_vocabulary_behavior = (
      token_model_flags.get_oov_token_behavior_from_command_line_flags())
  model_tokens = token_model_flags.get_tokens_from_command_line_flags(
      model_tokens=tokens.STRUCTURAL_TOKENS)
  main_function.run_gematria_model_from_command_line_flags(
      sequence_model_hlstm.HierarchicalLstmModel,
      use_deltas=_SEQ_2_SEQ.value,
      token_embedding_size=_TOKEN_EMBEDDING_SIZE.value,
      instruction_embedding_size=_INSTRUCTION_EMBEDDING_SIZE.value,
      block_embedding_size=_BLOCK_EMBEDDING_SIZE.value,
      bidirectional=_BIDIRECTIONAL.value,
      output_layers=flag_utils.layers_from_str(_OUTPUT_LAYERS.value),
      task_output_layers=flag_utils.layers_from_str(_TASK_OUTPUT_LAYERS.value),
      tokens=model_tokens,
      dtype=tf.dtypes.float32,
      out_of_vocabulary_behavior=out_of_vocabulary_behavior,
      out_of_vocabulary_injection_probability=(
          token_model_flags.OUT_OF_VOCABULARY_INJECTION_PROBABILITY.value),
  )


if __name__ == '__main__':
  tf.disable_v2_behavior()
  token_model_flags.mark_token_flags_as_required()
  token_model_flags.set_default_oov_replacement_token(tokens.UNKNOWN)
  app.run(main)
