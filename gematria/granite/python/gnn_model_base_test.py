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

import functools
from unittest import mock

from absl.testing import parameterized
from gematria.granite.python import gnn_model_base
from gematria.model.python import options
from gematria.testing.python import model_test
from gematria.model.python import model_blocks
import graph_nets
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf
import tf_keras


class TestGnnModel(gnn_model_base.GnnModelBase):
  """Trivial graph-based model for basic blocks.

  Each basic block is represented as chain of nodes, i.e. as a graph G=(V, E),
  where
    V = {v_1, ..., v_N},
    E = {(v_1, v_2), (v_2, v_3), ..., (v_N-1, v_N),
         (v_1, v_3), (v_2, v_4), ..., (v_N-2, v_N)},
  N = number of instructions in the basic block. The feature vector for each
  node, edge, and graph is a vector of ones or zeros of the right shape.

  The shape of the readout network depends on the mode:
   *  In the seq2num mode, the readout network computes the predictions from the
      global (graph) embedding vectors, by running them through a sequence of
      fully connected layers, followed by a single linear layer.
   *  In the seq2seq mode, the readout network does the same thing as above, but
      it's done for each node in the graph.

  This model is useless in practice, but it is strong enough to work well in our
  limited tests training on a single basic block.
  """

  def __init__(self, readout_dense_layers, readout_activation, **kwargs):
    """Initializes the test model.

    Args:
      readout_dense_layers: The shape of the readout network. This must be a
        non-empty sequence of ints; each value specifies the number of hidden
        units in one layer.
      readout_activation: The activation function used by all layers in the
        readout network. Any value accepted as 'activation' by
        tf_keras.layers.Dense() can be used here.
      **kwargs: Additional positional arguments are passed to the GnnModelBase
        constructor.
    """
    super().__init__(
        dtype=tf.dtypes.float32,
        node_feature_shape=(16,),
        edge_feature_shape=(16,),
        global_feature_shape=(16,),
        num_message_passing_iterations=8,
        **kwargs,
    )
    self.readout_dense_layers = []
    self.readout_activation = readout_activation
    for i, layer_size in enumerate(readout_dense_layers):
      self.readout_dense_layers.append(
          tf_keras.layers.Dense(
              layer_size,
              activation=self.readout_activation,
              name=f'dense_{i}',
              bias_initializer='glorot_normal',
          )
      )
    self.linear_layer = tf_keras.layers.Dense(
        self.num_tasks, activation='linear', use_bias=False
    )

  # @Override
  def _make_model_name(self):
    return 'TestGnnModel'

  # @Override
  def _start_batch(self):
    super()._start_batch()
    self._batch_networkxs = []

  # @Override
  def _make_batch_graphs_tuple(self):
    data_dicts = []
    for networkx in self._batch_networkxs:
      data_dicts.append(
          graph_nets.utils_np.networkx_to_data_dict(
              networkx,
              node_shape_hint=self._graph_node_feature_spec.shape,
              edge_shape_hint=self._graph_edge_feature_spec.shape,
              data_type_hint=self.numpy_dtype,
          )
      )
    return graph_nets.utils_tf.data_dicts_to_graphs_tuple(data_dicts)

  # @Override
  def _create_graph_network_modules(self):
    # We use a fully-connected network with the same shape in all contexts.
    mlp = functools.partial(
        snt.nets.MLP,
        output_sizes=(32, 16),
        w_init=tf_keras.initializers.glorot_normal(),
        b_init=tf_keras.initializers.glorot_normal(),
    )
    return (
        gnn_model_base.GraphNetworkLayer(
            module=graph_nets.modules.GraphNetwork(
                edge_model_fn=mlp, node_model_fn=mlp, global_model_fn=mlp
            ),
            num_iterations=None,
            layer_normalization=options.EnableFeature.NEVER,
            residual_connection=options.EnableFeature.NEVER,
        ),
    )

  # @Override
  def _execute_readout_network(self, graph_tuples, feed_dict) -> tf.Tensor:
    if self._use_deltas:
      dense_data = graph_tuples.nodes
    else:
      dense_data = graph_tuples.globals
    for layer in self.readout_dense_layers:
      dense_data = layer(dense_data)

    # Create a linear layer that computes a weighted sum of the output of the
    # last dense layer.
    return self.linear_layer(dense_data)

  # @Override
  def _add_basic_block_to_batch(self, block):
    graph = nx.DiGraph(
        features=np.zeros(
            self._graph_global_feature_spec.shape, dtype=self.numpy_dtype
        )
    )
    num_instructions = len(block.instructions)
    for i in range(num_instructions):
      graph.add_node(
          i,
          features=np.ones(
              self._graph_node_feature_spec.shape, dtype=self.numpy_dtype
          ),
      )
      if i > 0:
        graph.add_edge(
            i - 1,
            i,
            features=np.zeros(
                self._graph_edge_feature_spec.shape, dtype=self.numpy_dtype
            ),
        )
      if i > 1:
        graph.add_edge(
            i - 2,
            i,
            features=np.zeros(
                self._graph_edge_feature_spec.shape, dtype=self.numpy_dtype
            ),
        )
    self._batch_networkxs.append(graph)


class TestEncoderDecoderGnnModel(gnn_model_base.GnnModelBase):
  """A graph neural network model with a graph encoder and decoder.

  The translation of basic blocks is similar to how it is done in TestGnnModel,
  except that edge, node, and global features are all integer scalar values that
  are used as indices into node, edge, and global embedding lookup tables. The
  feature value of a node (edge, graph) is the order of its creation, starting
  with zero, modulo the number of node (edge, graph) tokens.

  The graph network consists of three modules (layers):
    1. An encoder module. It performs lookup in edge, node, and global embeding
       tables based on the features from the input graphs. The encoder module is
       applied once.
    2. A Graph network module, with multi-layer perceptron networks as update
       functions. This module is applied self._num_message_passing_iterations
       times.
    3. A decoder module. It runs a multi-layer perceptron network on each edge,
       node, and graph separately.

  The readout network runs a single linear layer (with bias) to sum up the
  output of the decoder.
  """

  def __init__(self, *, decoder_residual_connection=False, **kwargs):
    # The size of the embedding vectors. Each type of embedding vectors has a
    # different size to catch errors from mismatched tensors.
    self._node_embedding_size = 24
    self._edge_embedding_size = 8
    self._global_embedding_size = 12

    # The number of tokens for nodes, edges and graphs.
    self._num_node_types = 10
    self._num_edge_types = 14
    self._num_graph_types = 1

    self._decoder_residual_connection = decoder_residual_connection

    super().__init__(
        dtype=tf.dtypes.float32,
        # The embedding table lookup module expects the values to be integers.
        edge_feature_dtype=tf.dtypes.int32,
        node_feature_dtype=tf.dtypes.int32,
        global_feature_dtype=tf.dtypes.int32,
        # Node, edge, and global features are indices to the lookup table. We
        # need to make them scalars, so that the embedding lookup does not add
        # any extra dimensions.
        node_feature_shape=(),
        edge_feature_shape=(),
        global_feature_shape=(),
        num_message_passing_iterations=8,
        **kwargs,
    )

    self._linear_layer = tf_keras.layers.Dense(
        self.num_tasks, activation='linear'
    )

  # @Override
  def _make_model_name(self):
    return 'TestEncoderDecoderGnnModel'

  # @Override
  def _start_batch(self):
    super()._start_batch()
    self._batch_networkxs = []

  # @Override
  def _make_batch_graphs_tuple(self):
    data_dicts = []
    for networkx in self._batch_networkxs:
      data_dicts.append(
          graph_nets.utils_np.networkx_to_data_dict(
              networkx,
              node_shape_hint=self._graph_node_feature_spec.shape,
              edge_shape_hint=self._graph_edge_feature_spec.shape,
              data_type_hint=self.numpy_dtype,
          )
      )
    return graph_nets.utils_tf.data_dicts_to_graphs_tuple(data_dicts)

  # @Override
  def _create_graph_network_modules(self):
    mlp_initializers = {
        'w_init': tf_keras.initializers.glorot_normal(),
        'b_init': tf_keras.initializers.glorot_normal(),
    }
    embedding_initializer = tf_keras.initializers.glorot_normal()
    enable_decoder_residual_connection = (
        options.EnableFeature.ALWAYS
        if self._decoder_residual_connection
        else options.EnableFeature.NEVER
    )
    return (
        gnn_model_base.GraphNetworkLayer(
            module=graph_nets.modules.GraphIndependent(
                edge_model_fn=functools.partial(
                    snt.Embed,
                    vocab_size=self._num_edge_types,
                    embed_dim=self._edge_embedding_size,
                    initializer=embedding_initializer,
                ),
                node_model_fn=functools.partial(
                    snt.Embed,
                    vocab_size=self._num_node_types,
                    embed_dim=self._node_embedding_size,
                    initializer=embedding_initializer,
                ),
                global_model_fn=functools.partial(
                    snt.Embed,
                    vocab_size=self._num_graph_types,
                    embed_dim=self._global_embedding_size,
                    initializer=embedding_initializer,
                ),
                name='encoder',
            ),
            num_iterations=1,
            layer_normalization=options.EnableFeature.NEVER,
            residual_connection=options.EnableFeature.NEVER,
            edges_output_size=(None, self._edge_embedding_size),
            nodes_output_size=(None, self._node_embedding_size),
            globals_output_size=(None, self._global_embedding_size),
        ),
        gnn_model_base.GraphNetworkLayer(
            module=graph_nets.modules.GraphNetwork(
                edge_model_fn=functools.partial(
                    snt.nets.MLP,
                    output_sizes=(32, self._edge_embedding_size),
                    **mlp_initializers,
                ),
                node_model_fn=functools.partial(
                    snt.nets.MLP,
                    output_sizes=(32, self._node_embedding_size),
                    **mlp_initializers,
                ),
                global_model_fn=functools.partial(
                    snt.nets.MLP,
                    output_sizes=(32, self._global_embedding_size),
                    **mlp_initializers,
                ),
                name='graph_net',
            ),
            num_iterations=None,
            layer_normalization=options.EnableFeature.BY_FLAG,
            residual_connection=options.EnableFeature.BY_FLAG,
            edges_output_size=(None, self._edge_embedding_size),
            nodes_output_size=(None, self._node_embedding_size),
            globals_output_size=(None, self._global_embedding_size),
        ),
        gnn_model_base.GraphNetworkLayer(
            module=graph_nets.modules.GraphIndependent(
                edge_model_fn=functools.partial(
                    snt.nets.MLP,
                    output_sizes=(6,),
                    **mlp_initializers,
                ),
                node_model_fn=functools.partial(
                    snt.nets.MLP,
                    output_sizes=(5,),
                    **mlp_initializers,
                ),
                global_model_fn=functools.partial(
                    snt.nets.MLP,
                    output_sizes=(4,),
                    **mlp_initializers,
                ),
                name='decoder',
            ),
            num_iterations=1,
            layer_normalization=options.EnableFeature.ALWAYS,
            residual_connection=enable_decoder_residual_connection,
            edges_output_size=(None, 6),
            nodes_output_size=(None, 5),
            globals_output_size=(None, 4),
        ),
    )

  # @Override
  def _execute_readout_network(self, graphs_tuple, feed_dict) -> tf.Tensor:
    if self._use_deltas:
      # We add two nodes per instruction. To get one output row per instruction
      # as expected by the model in the seq2seq mode, we drop every other row
      # from the feature embedding vector.
      dense_data = graphs_tuple.nodes[::2]
    else:
      dense_data = graphs_tuple.globals
    # Create a linear layer that computes a weighted sum of the output of the
    # last dense layer.
    return self._linear_layer(dense_data)

  # @Override
  def _add_basic_block_to_batch(self, block):
    graph = nx.DiGraph(
        features=np.zeros(
            self._graph_global_feature_spec.shape,
            dtype=tf.dtypes.int32.as_numpy_dtype(),
        )
    )
    num_instructions = len(block.instructions)
    num_edges = 0
    for i in range(num_instructions):
      graph.add_node(
          i,
          features=np.array(
              i % self._num_node_types,
              dtype=self._graph_node_feature_spec.dtype.as_numpy_dtype(),
          ),
      )
      # NOTE(ondrasej): Make sure that we have at least one edge per batch, even
      # if the training sample contains just one instruction. This is necessary
      # because the layer normalization op crashes when running on an empty
      # tensor.
      graph.add_node(
          num_instructions + i,
          features=np.array(
              i % self._num_node_types,
              dtype=self._graph_node_feature_spec.dtype.as_numpy_dtype(),
          ),
      )
      graph.add_edge(
          num_instructions + i,
          i,
          features=np.array(
              num_edges % self._num_edge_types,
              dtype=self._graph_edge_feature_spec.dtype.as_numpy_dtype(),
          ),
      )
      num_edges += 1
      if i > 0:
        graph.add_edge(
            i - 1,
            i,
            features=np.array(
                num_edges % self._num_edge_types,
                dtype=self._graph_edge_feature_spec.dtype.as_numpy_dtype(),
            ),
        )
        num_edges += 1
      if i > 1:
        graph.add_edge(
            i - 2,
            i,
            features=np.array(
                num_edges % self._num_edge_types,
                dtype=self._graph_global_feature_spec.dtype.as_numpy_dtype(),
            ),
        )
        num_edges += 1
    self._batch_networkxs.append(graph)


class GnnModelBaseTest(parameterized.TestCase, model_test.TestCase):

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2num_single_task(self, loss_type, loss_normalization):
    model = TestGnnModel(
        readout_dense_layers=[32, 32],
        readout_activation='relu',
        loss_normalization=loss_normalization,
        loss_type=loss_type,
        learning_rate=0.01,
    )
    model.initialize()
    self.check_training_model(model)

  @parameterized.named_parameters(*model_test.OPTIMIZER_TYPES)
  def test_train_seq2num_single_task_optimizer_types(self, optimizer_type):
    model = TestGnnModel(
        readout_dense_layers=[32, 32],
        readout_activation='relu',
        loss_type=options.LossType.MEAN_SQUARED_ERROR,
        loss_normalization=options.ErrorNormalization.NONE,
        optimizer_type=optimizer_type,
        learning_rate=0.01,
    )
    model.initialize()
    self.check_training_model(model)

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2seq_single_task(self, loss_type, loss_normalization):
    model = TestGnnModel(
        readout_dense_layers=[32, 32],
        readout_activation='relu',
        loss_normalization=loss_normalization,
        loss_type=loss_type,
        use_deltas=True,
        learning_rate=0.01,
    )
    model.initialize()
    self.check_training_model(model)

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2num_encoder_decoder_model(
      self, loss_type, loss_normalization
  ):
    with mock.patch(
        'tf_keras.layers.LayerNormalization',
        side_effect=tf_keras.layers.LayerNormalization,
    ) as tf_keras_layer_norm:
      model = TestEncoderDecoderGnnModel(
          graph_module_layer_normalization=False,
          loss_normalization=loss_normalization,
          loss_type=loss_type,
          learning_rate=0.01,
      )

    model.initialize()
    self.assertEqual(
        tf_keras_layer_norm.call_args_list,
        [
            mock.call(name='graph_network_layer_norm_2_0_nodes'),
            mock.call(name='graph_network_layer_norm_2_0_edges'),
            mock.call(name='graph_network_layer_norm_2_0_globals'),
        ],
    )
    self.check_training_model(model)

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2seq_encoder_decoder_model(
      self, loss_type, loss_normalization
  ):
    with (
        mock.patch(
            'tf_keras.layers.LayerNormalization',
            side_effect=tf_keras.layers.LayerNormalization,
        ) as tf_keras_layer_norm,
        mock.patch(
            'gematria.model.python.model_blocks.ResidualConnectionLayer',
            side_effect=model_blocks.ResidualConnectionLayer,
        ) as residual_connection_layer,
    ):
      model = TestEncoderDecoderGnnModel(
          graph_module_layer_normalization=True,
          graph_module_residual_connections=False,
          loss_normalization=loss_normalization,
          loss_type=loss_type,
          use_deltas=True,
          learning_rate=0.01,
      )

    model.initialize()
    # NOTE(ondrasej): tf.math.add is called only when adding residual
    # connections. Since they are disabled in this test case, we should not see
    # any calls to this function.
    self.assertEqual(residual_connection_layer.call_args_list, [])
    self.assertEqual(
        tf_keras_layer_norm.call_args_list,
        [
            mock.call(name='graph_network_layer_norm_1_0_nodes'),
            mock.call(name='graph_network_layer_norm_1_0_edges'),
            mock.call(name='graph_network_layer_norm_1_0_globals'),
            mock.call(name='graph_network_layer_norm_1_1_nodes'),
            mock.call(name='graph_network_layer_norm_1_1_edges'),
            mock.call(name='graph_network_layer_norm_1_1_globals'),
            mock.call(name='graph_network_layer_norm_1_2_nodes'),
            mock.call(name='graph_network_layer_norm_1_2_edges'),
            mock.call(name='graph_network_layer_norm_1_2_globals'),
            mock.call(name='graph_network_layer_norm_1_3_nodes'),
            mock.call(name='graph_network_layer_norm_1_3_edges'),
            mock.call(name='graph_network_layer_norm_1_3_globals'),
            mock.call(name='graph_network_layer_norm_1_4_nodes'),
            mock.call(name='graph_network_layer_norm_1_4_edges'),
            mock.call(name='graph_network_layer_norm_1_4_globals'),
            mock.call(name='graph_network_layer_norm_1_5_nodes'),
            mock.call(name='graph_network_layer_norm_1_5_edges'),
            mock.call(name='graph_network_layer_norm_1_5_globals'),
            mock.call(name='graph_network_layer_norm_1_6_nodes'),
            mock.call(name='graph_network_layer_norm_1_6_edges'),
            mock.call(name='graph_network_layer_norm_1_6_globals'),
            mock.call(name='graph_network_layer_norm_1_7_nodes'),
            mock.call(name='graph_network_layer_norm_1_7_edges'),
            mock.call(name='graph_network_layer_norm_1_7_globals'),
            mock.call(name='graph_network_layer_norm_2_0_nodes'),
            mock.call(name='graph_network_layer_norm_2_0_edges'),
            mock.call(name='graph_network_layer_norm_2_0_globals'),
        ],
    )
    self.check_training_model(model)

  def test_train_seq2seq_model_with_residual_connections(self):
    with (
        mock.patch(
            'gematria.model.python.model_blocks.ResidualConnectionLayer',
            side_effect=model_blocks.ResidualConnectionLayer,
        ) as residual_connection_layer,
        mock.patch(
            'tf_keras.layers.Dense',
            side_effect=tf_keras.layers.Dense,
        ) as tf_keras_dense,
    ):
      model = TestEncoderDecoderGnnModel(
          graph_module_layer_normalization=True,
          graph_module_residual_connections=True,
          use_deltas=True,
          learning_rate=0.01,
      )

    model.initialize()
    self.assertEqual(
        residual_connection_layer.call_args_list,
        [
            mock.call(mock.ANY, name='residual_connection_1_0_nodes'),
            mock.call(mock.ANY, name='residual_connection_1_0_edges'),
            mock.call(mock.ANY, name='residual_connection_1_0_globals'),
            mock.call(mock.ANY, name='residual_connection_1_1_nodes'),
            mock.call(mock.ANY, name='residual_connection_1_1_edges'),
            mock.call(mock.ANY, name='residual_connection_1_1_globals'),
            mock.call(mock.ANY, name='residual_connection_1_2_nodes'),
            mock.call(mock.ANY, name='residual_connection_1_2_edges'),
            mock.call(mock.ANY, name='residual_connection_1_2_globals'),
            mock.call(mock.ANY, name='residual_connection_1_3_nodes'),
            mock.call(mock.ANY, name='residual_connection_1_3_edges'),
            mock.call(mock.ANY, name='residual_connection_1_3_globals'),
            mock.call(mock.ANY, name='residual_connection_1_4_nodes'),
            mock.call(mock.ANY, name='residual_connection_1_4_edges'),
            mock.call(mock.ANY, name='residual_connection_1_4_globals'),
            mock.call(mock.ANY, name='residual_connection_1_5_nodes'),
            mock.call(mock.ANY, name='residual_connection_1_5_edges'),
            mock.call(mock.ANY, name='residual_connection_1_5_globals'),
            mock.call(mock.ANY, name='residual_connection_1_6_nodes'),
            mock.call(mock.ANY, name='residual_connection_1_6_edges'),
            mock.call(mock.ANY, name='residual_connection_1_6_globals'),
            mock.call(mock.ANY, name='residual_connection_1_7_nodes'),
            mock.call(mock.ANY, name='residual_connection_1_7_edges'),
            mock.call(mock.ANY, name='residual_connection_1_7_globals'),
        ],
    )
    self.assertEqual(
        tf_keras_dense.call_args_list, [mock.call(1, activation='linear')]
    )
    self.check_training_model(model)

  def test_train_seq2seq_model_with_residual_connections_with_linear_transform(
      self,
  ):
    with (
        mock.patch(
            'gematria.model.python.model_blocks.ResidualConnectionLayer',
            side_effect=model_blocks.ResidualConnectionLayer,
        ) as residual_connection_layer,
        mock.patch(
            'tf_keras.layers.Dense',
            side_effect=tf_keras.layers.Dense,
        ) as tf_keras_dense,
    ):
      model = TestEncoderDecoderGnnModel(
          graph_module_layer_normalization=False,
          graph_module_residual_connections=False,
          decoder_residual_connection=True,
          use_deltas=True,
          learning_rate=0.01,
      )
    model.initialize()
    self.assertEqual(
        residual_connection_layer.call_args_list,
        [
            mock.call(mock.ANY, name='residual_connection_2_0_nodes'),
            mock.call(mock.ANY, name='residual_connection_2_0_edges'),
            mock.call(mock.ANY, name='residual_connection_2_0_globals'),
        ],
    )
    self.assertEqual(
        tf_keras_dense.call_args_list,
        [
            mock.call(
                activation=tf_keras.activations.linear,
                name='residual_connection_2_0_nodes_transformation',
                units=5,
                use_bias=False,
            ),
            mock.call(
                activation=tf_keras.activations.linear,
                name='residual_connection_2_0_edges_transformation',
                units=6,
                use_bias=False,
            ),
            mock.call(
                activation=tf_keras.activations.linear,
                name='residual_connection_2_0_globals_transformation',
                units=4,
                use_bias=False,
            ),
            mock.call(1, activation='linear'),
        ],
    )
    self.check_training_model(model)

  # TODO(ondrasej): Add tests for multi-task learning with GNNs.
  # TODO(ondrasej): Add explicit tests for inference.


if __name__ == '__main__':
  tf.test.main()
