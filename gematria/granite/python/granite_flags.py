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
"""Defines the common flags for models."""

from absl import flags

from gematria.model.python import options
from gematria.utils.python import flag_utils

NODE_EMBEDDING_SIZE = flags.DEFINE_integer(
    'gematria_node_embedding_size',
    128,
    (
        'The size of node embedding vectors used in the graph network. This is'
        ' both the size of the learned embedding vectors and the size of the'
        ' feature vectors used during graph network computation.'
    ),
)
EDGE_EMBEDDING_SIZE = flags.DEFINE_integer(
    'gematria_edge_embedding_size',
    128,
    (
        'The size of edge embedding vectors used in the graph network. This is'
        ' both the size of the learned embedding vectors and the size of the'
        ' feature vectors used during graph network computation.'
    ),
)
GLOBAL_EMBEDDING_SIZE = flags.DEFINE_integer(
    'gematria_global_embedding_size',
    128,
    'The size of global embedding vectors used in the graph network.',
)

NODE_UPDATE_LAYERS = flags.DEFINE_string(
    'gematria_node_update_layers',
    '',
    (
        'Specifies a comma-separated string that represents the number of dense'
        ' layers and their sizes used to update the node feature vectors in'
        ' each step of the graph neural network computation. An additional'
        ' dense layer of size --gematria_node_embedding_size is added'
        ' automatically after the last layer.'
    ),
)
EDGE_UPDATE_LAYERS = flags.DEFINE_string(
    'gematria_edge_update_layers',
    '',
    (
        'Specifies a comma-separated string that represents the number of dense'
        ' layers and their sizes used to update the'
        ' edge feature vectors in each step of the graph neural network'
        ' computation. An additional dense layer of size'
        ' --gematria_edge_embedding_size is added automatically after the last'
        ' layer.'
    ),
)
GLOBAL_UPDATE_LAYERS = flags.DEFINE_string(
    'gematria_global_update_layers',
    '',
    (
        'Specifies a comma-separated string that represents the number of dense'
        ' layers and their sizes used to update the global feature vectors in'
        ' each step of the graph neural network computation. An additional'
        ' dense layer of size --gematria_global_embedding_size is added'
        ' automatically after the last layer.'
    ),
)

USE_SENT_EDGES = flags.DEFINE_bool(
    'gematria_use_sent_edges',
    False,
    (
        'Specifies whether the Node block of the graph network should use the'
        ' sent edges as a part of its input. This allows information to flow in'
        ' both directions and in particular to propagate information from'
        ' output operands back to the instructions writing to them.'
    ),
)

GRAPH_MODULE_RESIDUAL_CONNECTIONS = flags.DEFINE_bool(
    'gematria_graph_module_residual_connections',
    False,
    (
        "Use residual connections around graph network modules' computation of"
        ' node, edge, and global feature vectors.'
    ),
)
READOUT_RESIDUAL_CONNECTIONS = flags.DEFINE_bool(
    'gematria_readout_residual_connections',
    False,
    'Add a residual connection for the shared part of the readout network.',
)
TASK_READOUT_RESIDUAL_CONNECTIONS = flags.DEFINE_bool(
    'gematria_task_readout_residual_connections',
    False,
    (
        'Add a residual connection for the task-specific part of the readout'
        ' network.'
    ),
)

RNN_OUTPUT_SIZE = flags.DEFINE_integer(
    'gematria_rnn_output_size',
    128,
    (
        'The size of the output vectors of the RNN network that is passed to'
        ' the readout network. The actual size of the state passed between the'
        ' RNN cells is twice this value.'
    ),
)
RNN_BIDIRECTIONAL = flags.DEFINE_bool(
    'gematria_rnn_bidirectional',
    False,
    'Specifies whether a bidirectional RNN is used after graph neural network.',
)
RNN_DROPOUT = flags.DEFINE_float(
    'gematria_rnn_dropout',
    0.0,
    (
        'A float between 0 and 1. Fraction of units to drop for the linear'
        ' transformation of the inputs.'
    ),
)
RNN_TYPE = flags.DEFINE_enum_class(
    'gematria_rnn_type',
    default=options.RnnType.NONE,
    enum_class=options.RnnType,
    help='Specifies the RNN cell type to be used after GNN.',
)

READOUT_LAYERS = flags.DEFINE_string(
    'gematria_readout_layers',
    '',
    (
        'Specifies a comma-separated string that represents the number of'
        ' shared dense layers and their sizes used to compute the output of the'
        ' network. These dense layers consume the outputs of the graph network.'
        ' When empty, the outputs of the graph network are passed directly to'
        ' the input of the task-specific dense layers.'
    ),
)
TASK_READOUT_LAYERS = flags.DEFINE_string(
    'gematria_task_readout_layers',
    '',
    (
        'Specifies a comma-separated string that represents the number of '
        ' task-specific dense layers and their sizes used to compute the output'
        ' of the network. The task-specific dense layers consume the output of'
        ' the shared dense layers; they are followed by a single task-specific'
        ' linear layer that produces the output of the network for the task.'
        ' When empty, the linear layer is added directly after the shared dense'
        ' layers or after the graph network.'
    ),
)

USE_DELTAS = flags.DEFINE_bool(
    'gematria_seq2seq',
    False,
    (
        'When True, the prediction for a basic block is computed as a sum of'
        ' per-instruction predictions computed from feature of nodes'
        ' corresponding to instructions in the basic block graphs. Otherwise,'
        ' the predictions are computed from the global feature vector of each'
        ' basic block graph.'
    ),
)
NUM_MESSAGE_PASSING_ITERATIONS = flags.DEFINE_integer(
    'gematria_num_message_passing_iterations',
    8,
    'The number of iterations of the main graph network module.',
)

GRAPH_MODULE_LAYER_NORMALIZATION = flags.DEFINE_bool(
    'gematria_graph_module_layer_normalization',
    True,
    (
        'Inserts a layer normalization step at the output of each iteration of'
        ' the graph network module.'
    ),
)
READOUT_INPUT_LAYER_NORMALIZATION = flags.DEFINE_bool(
    'gematria_readout_input_layer_normalization',
    True,
    (
        'Insert a layer normalization step before the input of the shared'
        ' readout network. Note that this is redundant when using'
        ' --gematria_graph_module_layer_normalization.'
    ),
)
TASK_READOUT_INPUT_LAYER_NORMALIZATION = flags.DEFINE_bool(
    'gematria_task_readout_input_layer_normalization',
    True,
    (
        'Inserts a layer normalization step before the input of the'
        ' task-specific readout network.'
    ),
)

_MUST_BE_POSITIVE_WITH_RNN = (
    'RNN size must be positive when RNN type is not NONE.'
)

flags.register_validator(
    NODE_UPDATE_LAYERS.name,
    flag_utils.is_positive_integer_list,
    flag_utils.MUST_BE_POSITIVE_INTEGER_LIST_ERROR,
)
flags.register_validator(
    EDGE_UPDATE_LAYERS.name,
    flag_utils.is_positive_integer_list,
    flag_utils.MUST_BE_POSITIVE_INTEGER_LIST_ERROR,
)
flags.register_validator(
    GLOBAL_UPDATE_LAYERS.name,
    flag_utils.is_positive_integer_list,
    flag_utils.MUST_BE_POSITIVE_INTEGER_LIST_ERROR,
)
flags.register_validator(
    READOUT_LAYERS.name,
    flag_utils.is_positive_integer_list,
    flag_utils.MUST_BE_POSITIVE_INTEGER_LIST_ERROR,
)
flags.register_validator(
    TASK_READOUT_LAYERS.name,
    flag_utils.is_positive_integer_list,
    flag_utils.MUST_BE_POSITIVE_INTEGER_LIST_ERROR,
)


@flags.multi_flags_validator(
    [RNN_TYPE.name, RNN_OUTPUT_SIZE.name], message=_MUST_BE_POSITIVE_WITH_RNN
)
def _is_rnn_size_positive(flags_dict):
  return (
      flags_dict[RNN_TYPE.name] == options.RnnType.NONE
      or flags_dict[RNN_OUTPUT_SIZE.name] > 0
  )


flags.register_validator(
    NODE_EMBEDDING_SIZE.name,
    flag_utils.is_positive,
    flag_utils.MUST_BE_POSITIVE_ERROR,
)
flags.register_validator(
    EDGE_EMBEDDING_SIZE.name,
    flag_utils.is_positive,
    flag_utils.MUST_BE_POSITIVE_ERROR,
)
flags.register_validator(
    GLOBAL_EMBEDDING_SIZE.name,
    flag_utils.is_positive,
    flag_utils.MUST_BE_POSITIVE_ERROR,
)
flags.register_validator(
    NUM_MESSAGE_PASSING_ITERATIONS.name,
    flag_utils.is_positive,
    flag_utils.MUST_BE_POSITIVE_ERROR,
)

flags.register_validator(
    RNN_DROPOUT.name,
    flag_utils.is_between_zero_and_one,
    flag_utils.MUST_BE_BETWEEN_ZERO_AND_ONE,
)
