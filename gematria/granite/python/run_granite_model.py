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
"""Main function for the token-based Granite model."""

from absl import app
from gematria.basic_block.python import tokens
from gematria.granite.python import granite_flags
from gematria.granite.python import rnn_token_model
from gematria.granite.python import token_graph_builder_model
from gematria.model.python import main_function
from gematria.model.python import options
from gematria.model.python import token_model_flags
from gematria.utils.python import flag_utils
import tensorflow.compat.v1 as tf


def main(argv):
  del argv  # Unused
  if granite_flags.RNN_TYPE.value == options.RnnType.NONE:
    model_class = token_graph_builder_model.TokenGraphBuilderModel
    rnn_kwargs = {}
  else:
    model_class = rnn_token_model.RnnTokenModel
    rnn_kwargs = {
        'rnn_output_size': granite_flags.RNN_OUTPUT_SIZE.value,
        'rnn_dropout': granite_flags.RNN_DROPOUT.value,
        'rnn_bidirectional': granite_flags.RNN_BIDIRECTIONAL.value,
        'rnn_type': granite_flags.RNN_TYPE.value,
    }
  out_of_vocabulary_behavior = (
      token_model_flags.get_oov_token_behavior_from_command_line_flags()
  )
  model_tokens = token_model_flags.get_tokens_from_command_line_flags(
      model_tokens=tokens.STRUCTURAL_TOKENS
  )
  model_annotation_names = (
      token_model_flags.get_annotation_names_from_command_line_flags()
  )

  main_function.run_gematria_model_from_command_line_flags(
      model_class,
      tokens=model_tokens,
      immediate_token=tokens.IMMEDIATE,
      fp_immediate_token=tokens.IMMEDIATE,
      address_token=tokens.ADDRESS,
      memory_token=tokens.MEMORY,
      annotation_names=model_annotation_names,
      dtype=tf.dtypes.float32,
      node_embedding_size=granite_flags.NODE_EMBEDDING_SIZE.value,
      edge_embedding_size=granite_flags.EDGE_EMBEDDING_SIZE.value,
      global_embedding_size=granite_flags.GLOBAL_EMBEDDING_SIZE.value,
      node_update_layers=flag_utils.layers_from_str(
          granite_flags.NODE_UPDATE_LAYERS.value
      ),
      edge_update_layers=flag_utils.layers_from_str(
          granite_flags.EDGE_UPDATE_LAYERS.value
      ),
      global_update_layers=flag_utils.layers_from_str(
          granite_flags.GLOBAL_UPDATE_LAYERS.value
      ),
      readout_layers=flag_utils.layers_from_str(
          granite_flags.READOUT_LAYERS.value
      ),
      task_readout_layers=flag_utils.layers_from_str(
          granite_flags.TASK_READOUT_LAYERS.value
      ),
      graph_module_layer_normalization=(
          granite_flags.GRAPH_MODULE_LAYER_NORMALIZATION.value
      ),
      readout_input_layer_normalization=(
          granite_flags.READOUT_INPUT_LAYER_NORMALIZATION.value
      ),
      task_readout_input_layer_normalization=(
          granite_flags.TASK_READOUT_INPUT_LAYER_NORMALIZATION.value
      ),
      graph_module_residual_connections=(
          granite_flags.GRAPH_MODULE_RESIDUAL_CONNECTIONS.value
      ),
      readout_residual_connections=(
          granite_flags.READOUT_RESIDUAL_CONNECTIONS.value
      ),
      task_readout_residual_connections=(
          granite_flags.TASK_READOUT_RESIDUAL_CONNECTIONS.value
      ),
      use_sent_edges=granite_flags.USE_SENT_EDGES.value,
      out_of_vocabulary_behavior=out_of_vocabulary_behavior,
      out_of_vocabulary_injection_probability=(
          token_model_flags.OUT_OF_VOCABULARY_INJECTION_PROBABILITY.value
      ),
      num_message_passing_iterations=(
          granite_flags.NUM_MESSAGE_PASSING_ITERATIONS.value
      ),
      use_deltas=granite_flags.USE_DELTAS.value,
      **rnn_kwargs,
  )


if __name__ == '__main__':
  token_model_flags.mark_token_flags_as_required()
  token_model_flags.set_default_oov_replacement_token(tokens.UNKNOWN)
  tf.disable_v2_behavior()
  app.run(main)
