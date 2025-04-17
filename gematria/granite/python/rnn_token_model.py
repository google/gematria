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
"""A token-based Granite model with an RNN layer for reducing node embeddings.

Provides the RnnTokenModel class. See the documentation of the class for more
information about the model.
"""

from gematria.granite.python import token_graph_builder_model
from gematria.model.python import options
import tensorflow.compat.v1 as tf
import tensorflow as tf2
import tf_keras

_RNN_TYPE_TO_TF = {'LSTM': tf_keras.layers.LSTM, 'GRU': tf_keras.layers.GRU}


class RnnTokenModel(token_graph_builder_model.TokenGraphBuilderModel):
  """A token-based model using the Granite graph construction and an RNN layer.

  The model uses a graph network to compute embedding vectors for instructions
  and an RNN layer to process these vectors using short- and long-term
  relations between instructions. The graph network is the same as in the
  TokenGraphBuilderModel class in
  gematria/granite/python/token_graph_builder_model.py.

  The model is composed of the following parts:
    1. Token-based embedding vectors. Each token recognized by
       feature_tokens.TokenFeatureFactory is assigned a learnable embedding
       vector of a given size.
    2. A Graph neural network using a sequence of dense layers as update
       functions.
    3. An RNN network that consumes the feature vectors of instruction nodes
       from the graph neural net. The instruction nodes are presented to the
       RNN network in the order in which the instructions appear in the basic
       block.
    4. An optional shared dense readout network that processes the outputs of
       the RNN layer. When this network is not used, the data is passed
       unchanged to the next step.
    5. An optional task-specific dense readout network that processes the
       outputs of the shared readout network. When this network is not used, the
       data is passed unchanged to the next step.
    6. A task-specific linear layer that computes the output of the network.

  The behavior of the dense readout networks is different for seq2seq and
  seq2num models:
    * For seq2num models, we use the output of the last RNN cell as an input
      for the dense readout network, i.e. the readout is computed once per basic
      block.
    * For seq2seq models, we apply the readout network on the output of each
      RNN cell, and the readout network predicts the delta (the throughput
      increment) for the instruction corresponding to that RNN cell. The
      prediction for the basic block is then computed as the sum of the
      per-instruction predictions.
  """

  # The size of the output of the RNN network. Note that this does not include
  # the hidden state passed between consecutive two RNN cells.
  _rnn_output_size: int
  # A float number that specifies the fraction of inputs that are for
  # the linear transformation of the inputs.
  _rnn_dropout: float
  # Specifies whether to use a bidirectional RNN after graph neural network.
  _rnn_bidirectional: bool
  # Specifies the type of RNN (e.g. LSTM, GRU) that is used after GNN.
  _rnn_type: options.RnnType

  def __init__(
      self,
      rnn_output_size: int,
      rnn_dropout: float,
      rnn_bidirectional: bool,
      rnn_type: options.RnnType,
      **kwargs,
  ) -> None:
    """Initializes the token-based model with the RNN layer.

    Args:
      rnn_output_size: The requested size of the output of the RNN cells used to
        compute the basic block embedding vector. Note that this size does not
        include the hidden state vector passed between RNN cells.
      rnn_dropout: A float between zero and one. Fraction of the units to drop
        for the linear transformation of the inputs.
      rnn_bidirectional: Specifies whether a bidirectional RNN is used after
        graph neural network.
      rnn_type: Specifies the type of RNN (e.g. LSTM, GRU) that is used after
        GNN.
      **kwargs: All other (keyword) arguments are passed to the constructor of
        the base class.
    """
    # NOTE(ondrasej): We need self._delta_block_index_tensor to create inputs
    # for the RNN layer even when running in a seq2num mode.
    kwargs['create_delta_block_index'] = True
    super().__init__(**kwargs)
    self._rnn_output_size = rnn_output_size
    self._rnn_dropout = rnn_dropout
    self._rnn_bidirectional = rnn_bidirectional
    self._rnn_type = rnn_type

  def _make_model_name(self) -> str:
    base_name = super()._make_model_name()
    return (
        f'Rnn{base_name},rnn_state_size={self._rnn_output_size}'
        f',rnn_dropout={self._rnn_dropout}'
        f',bidirectional={self._rnn_bidirectional}'
    )

  def initialize(self) -> None:
    super().initialize()
    if self._readout_input_layer_normalization:
      self._rnn_layer_normalization = tf_keras.layers.LayerNormalization()
    # TODO(ayazdan): Figure out how to pass `training` flag to the pipeline.
    self._rnn_layer = _RNN_TYPE_TO_TF[self._rnn_type.name](
        self._rnn_output_size,
        dropout=self._rnn_dropout,
        return_sequences=self._use_deltas,
    )
    if self._rnn_bidirectional:
      self._rnn_layer = tf_keras.layers.Bidirectional(self._rnn_layer)

  def _execute_readout_network(self, graph_tuple, feed_dict) -> tf.Tensor:
    instruction_features = tf2.boolean_mask(
        graph_tuple.nodes, feed_dict['instruction_node_mask']
    )

    # Normalize the instruction features if needed.
    if self._readout_input_layer_normalization:
      instruction_features = self._rnn_layer_normalization(instruction_features)

    # A ragged tensor that contains the basic blocks in the batch. Each element
    # of the ragged tensor corresponds to one basic blocks in the batch, and it
    # contains a sequence of feature vectors of the instructions in the basic
    # block.
    blocks_ragged = tf2.RaggedTensor.from_value_rowids(
        instruction_features, feed_dict['delta_block_index']
    )

    # Depending on the value of self._use_deltas:
    #  * In the seq2num mode (self._use_deltas == False), rnn_outputs contains
    #    one state vector for each basic block (the output of the last RNN cell
    #    for that basic block).
    #  * In the seq2seq mode (self._use_deltas == True), it is a ragged vector
    #    in the same format as blocks_ragged, and for each instruction we have
    #    the output of the RNN cell at the corresponding position.
    rnn_outputs = self._rnn_layer(blocks_ragged)

    if self._use_deltas:
      # In seq2seq mode, convert the ragged tensor back to a normal tensor that
      # we can process as a tensor of deltas. We do this by concatenating all
      # the sequences from the ragged tensor.
      rnn_outputs = rnn_outputs.values

    # We apply the readout network on the outputs of the RNN. While this has
    # different semantic in seq2seq vs seq2num modes, the network has exactly
    # the same structure. The outputs of the RNN network are already in the
    # (-1, 1) range, so we skip any additional normalization steps.
    return self._execute_dense_readout_network(rnn_outputs)
