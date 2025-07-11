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
"""Implementation of the GRANITE model."""

import functools
import itertools
from typing import Callable, Optional, Sequence

from absl import logging
from gematria.granite.python import gnn_model_base
from gematria.granite.python import graph_builder
from gematria.granite.python import graph_builder_model_base
from gematria.model.python import model_blocks
from gematria.model.python import options
import graph_nets
import sonnet as snt
import tensorflow as tf
import tf_keras


class TokenGraphBuilderModel(graph_builder_model_base.GraphBuilderModelBase):
  """A token based model that uses a graph network and a dense readout network.

  The model is composed of three parts:
    1. Token-based embedding vectors. Each token recognized by the
       BasicBlockGraphBuilder is assigned a learnable embedding vector of
       a given size.
    2. A Graph neural network based on graph_nets.modules.GraphNetwork using a
       sequence of dense layers as update functions.
    3. An optional shared readout network operating on individual instruction
       embedding vectors or on graph embedding vectors.
    4. An optional task-specific readout network operating on the outputs of the
       shared readout network.
    5. A task-specific linear layer that computes the output of the network.

  The number of layers and their sizes can be tuned for each update function and
  for the readout layer separately.
  """

  READOUT_VARIABLES = 'TokenGraphBuilderModel.readout'
  TASK_READOUT_VARIABLES = 'TokenGraphBuilderModel.task_readout'

  def __init__(
      self,
      node_embedding_size: int,
      edge_embedding_size: int,
      global_embedding_size: int,
      node_update_layers: Sequence[int],
      edge_update_layers: Sequence[int],
      global_update_layers: Sequence[int],
      readout_layers: Sequence[int],
      task_readout_layers: Sequence[int],
      readout_input_layer_normalization: bool = True,
      task_readout_input_layer_normalization: bool = True,
      readout_residual_connections: bool = False,
      task_readout_residual_connections: bool = False,
      use_sent_edges: bool = False,
      update_activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
      readout_activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
      **kwargs,
  ) -> None:
    """Initializes the model.

    Args:
      node_embedding_size: The size of the embedding vectors used for the nodes
        of the basic block graphs.
      edge_embedding_size: The size of the embedding vectors used for the edges
        of the basic block graphs.
      global_embedding_size: The size of the embedding vectors used for the
        graph feature vectors of the basic block graphs.
      node_update_layers: The sizes of the dense layers in the node embedding
        vector update function. The layer at index 0 comes right after the
        input; an additional layer of size `node_embedding_size` is inserted
        automatically at the end.
      edge_update_layers: The sizes of the dense layers in the edge embedding
        vector update function. The layer at index 0 comes right after the
        input; an additional layer of size `edge_embedding_size` is inserted
        automatically at the end.
      global_update_layers: The sizes of the dense layers in the global (graph)
        embedding vector update function. The layer at index 0 comes right after
        the input; an additional layer of size `global_embedding_size` is
        inserted automatically at the end.
      readout_layers: The sizes of the shared dense layers in the readout
        network. The layer at index 0 comes right after the output of the graph
        network.
      task_readout_layers: The sizes of the task-specific dense layers in the
        readout network. There is one such network for each task learned by the
        model. The layer at index 0 comes right after the last shared readout
        layer; an additional linear layer with no bias and one outputs is
        inserted automatically at the end. The output of this final linear layer
        corresponds to the output for the task.
      readout_input_layer_normalization: When True, layer normalization is
        applied to the input of the shared dense readout network. Note that with
        graph_module_layer_normalization and non-zero number of iterations of
        the graph module, the input of the readout network is already normalized
        by the last graph module.
      task_readout_input_layer_normalization: When True, layer normalization is
        applied to the input of the per-task dense readout networks.
      readout_residual_connections: When True, a residual connection is added
        around the shared part of the readout network.
      task_readout_residual_connections: When True, a residual connection is
        added around the task-specific part of the readout network.
      use_sent_edges: When True, the main graph network module uses information
        from sent edges in addition to information from received edges when
        updating the node feature vectors.
      update_activation: The activation function used by the dense networks used
        to update the node, edge, and global feature vectors. When not
        specified, the leaky ReLU function is used.
      readout_activation: The activation function used by the dense readout
        network. When not specified, the leaky ReLU function is used.
      **kwargs: All remaining keyword arguments are passed to the constructor of
        the base class.
    """
    super().__init__(**kwargs)
    self._readout_layers = tuple(readout_layers)
    self._task_readout_layers = tuple(task_readout_layers)
    self._node_embedding_size = node_embedding_size
    self._edge_embedding_size = edge_embedding_size
    self._global_embedding_size = global_embedding_size

    self._use_sent_edges = use_sent_edges

    self._node_update_layers = tuple(
        itertools.chain(node_update_layers, (node_embedding_size,))
    )
    self._edge_update_layers = tuple(
        itertools.chain(edge_update_layers, (edge_embedding_size,))
    )
    self._global_update_layers = tuple(
        itertools.chain(global_update_layers, (global_embedding_size,))
    )

    self._readout_input_layer_normalization = readout_input_layer_normalization
    self._task_readout_input_layer_normalization = (
        task_readout_input_layer_normalization
    )

    self._readout_residual_connections = readout_residual_connections
    self._task_readout_residual_connections = task_readout_residual_connections

    # NOTE(ondrasej): This just creates a closure around the constructor, but it
    # doesn't make any TensorFlow ops until this closure is called, i.e. it is
    # cheap to create or discard this object.
    leaky_relu = functools.partial(tf_keras.activations.relu, alpha=0.1)
    self._readout_activation = readout_activation or leaky_relu
    self._update_activation = update_activation or leaky_relu

    if self._num_annotations > self._node_embedding_size:
      raise ValueError(
          '_num_annotations cannot be greater than _node_embedding_size.'
      )

    # The length of the learnt part of the instruction node embedding vectors.
    # The remaining elements are filled in with instruction annotations.
    self._common_node_embedding_size = (
        self._node_embedding_size - self._num_annotations
    )

  def initialize(self):
    super().initialize()
    self._create_dense_readout_network()

  # @Override
  def _make_model_name(self) -> str:
    # TODO(ondrasej): Use a string provided by the token feature factory as the
    # name of the feature provider.
    return (
        'TokenGraphBuilderModel: '
        f'mp_iterations={self._num_message_passing_iterations} '
        f'node_embedding_size={self._node_embedding_size}, '
        f'node_update_layers={self._node_update_layers!r}, '
        f'edge_embedding_size={self._edge_embedding_size}, '
        f'edge_update_layers={self._edge_update_layers!r}, '
        f'global_embedding_size={self._global_embedding_size}, '
        f'global_update_layers={self._global_update_layers!r}, '
        f'readout_activation={self._readout_activation}, '
        f'readout_layers={self._readout_layers!r}, '
        f'task_readout_layers={self._task_readout_layers!r}, '
        f'graph_module_layer_norm={self._graph_module_layer_normalization}, '
        f'readout_input_layer_norm={self._readout_input_layer_normalization}, '
        'task_readout_input_layer_norm='
        f'{self._task_readout_input_layer_normalization}'
    )

  def _create_dense_readout_network(self):
    self._dense_readout_layers = []
    for size in self._readout_layers:
      self._dense_readout_layers.append(
          tf_keras.layers.Dense(
              size,
              activation=self._readout_activation,
              bias_initializer='glorot_normal',
          )
      )
    self._task_dense_sum_layers = []
    self._task_dense_readout_layers = []
    for _ in range(self.num_tasks):
      for size in self._task_readout_layers:
        self._task_dense_readout_layers.append(
            tf_keras.layers.Dense(
                size,
                activation=self._readout_activation,
                bias_initializer='glorot_normal',
            )
        )
      self._task_dense_sum_layers.append(
          tf_keras.layers.Dense(
              1, activation=tf_keras.activations.linear, use_bias=False
          )
      )
    if self._readout_input_layer_normalization:
      self._readout_layer_normalization_input = (
          tf_keras.layers.LayerNormalization(
              name='readout_input_layer_normalization'
          )
      )
    if self._task_readout_input_layer_normalization:
      self._readout_layer_normalization = tf_keras.layers.LayerNormalization(
          name='task_readout_input_layer_normalization'
      )
    residual_size = self._global_embedding_size
    if self._use_deltas:
      residual_size = self._node_embedding_size
    if self._readout_residual_connections:
      if self._readout_layers:
        self._readout_residual_layer = model_blocks.ResidualConnectionLayer(
            name='readout_residual_connections',
            layer_input_shapes=(
                tf.TensorShape([None, self._readout_layers[-1]]),
                tf.TensorShape([None, residual_size]),
            ),
        )
    self._task_residual_layers = []
    for task_index in range(self.num_tasks):
      if self._task_readout_residual_connections:
        if self._task_readout_layers:
          self._task_residual_layers.append(
              model_blocks.ResidualConnectionLayer(
                  name='task_readout_residual_connections',
                  layer_input_shapes=(
                      tf.TensorShape([None, self._task_readout_layers[-1]]),
                      tf.TensorShape([None, self._readout_layers[-1]]),
                  ),
              )
          )

  def _execute_dense_readout_network(self, data: tf.Tensor) -> tf.Tensor:
    """Creates the dense part of the readout network from `data`.

    Args:
      data: A tensor that contains the outputs of the graph network. This tensor
        is passed as an input to the dense network.

    Returns:
      A tensor of shape (None, num_tasks) that contains the output of the dense
      readout networks for all tasks.
    """
    readout_input = data
    for dense in self._dense_readout_layers:
      data = dense(data)
    if self._readout_residual_connections:
      if self._readout_layers:
        data = self._readout_residual_layer((data, readout_input))
      else:
        logging.warning(
            'Readout residual connections are enabled, but the'
            ' readout network has no layers.'
        )
    if self._task_readout_input_layer_normalization:
      data = self._readout_layer_normalization(data)
    task_outputs = []
    for task_index in range(self.num_tasks):
      task_data = data
      for readout_index, size in enumerate(self._task_readout_layers):
        dense = self._task_dense_readout_layers[
            task_index * len(self._task_readout_layers) + readout_index
        ]
        task_data = dense(task_data)
      if self._task_readout_residual_connections:
        if self._task_readout_layers:
          task_data = self._task_residual_layers[task_index]((task_data, data))
        else:
          logging.warning(
              'Task readout residual connections are enabled, but'
              ' the task readout network has no layers.'
          )
      # Create a linear layer that computes a weighted sum of the output of the
      # last dense layer.
      linear_layer = self._task_dense_sum_layers[task_index]
      task_outputs.append(linear_layer(task_data))
    return tf.concat(task_outputs, axis=1)

  def _execute_readout_network(self, graph_tuple, feed_dict) -> tf.Tensor:
    if self._use_deltas:
      data = tf.boolean_mask(
          graph_tuple.nodes, feed_dict['instruction_node_mask']
      )
    else:
      data = graph_tuple.globals
    if self._readout_input_layer_normalization:
      layer_normalization = self._readout_layer_normalization_input
      data = layer_normalization(data)
    return self._execute_dense_readout_network(data)

  def _create_graph_network_modules(
      self,
  ) -> Sequence[gnn_model_base.GraphNetworkLayer]:
    # Disable module-attr around uses of tf_keras.initializers.glorot_normal as
    # some tf_keras installations setup what is available in tf_keras.initializers
    # using logic in __init__.py that pytype is not able to understand
    # correctly.
    # pytype: disable=module-attr
    mlp_initializers = {
        'w_init': tf_keras.initializers.glorot_normal(),
        'b_init': tf_keras.initializers.glorot_normal(),
    }
    embedding_initializer = tf_keras.initializers.glorot_normal()
    # pytype: enable=module-attr
    return (
        gnn_model_base.GraphNetworkLayer(
            module=graph_nets.modules.GraphIndependent(
                edge_model_fn=functools.partial(
                    snt.Embed,
                    # TODO(ondrasej): Pybind11 generated enum types do not
                    # implement the full Python enum interface. Replace this
                    # with len(graph_builder.EdgeType) when
                    # https://github.com/pybind/pybind11/issues/2332 is fixed.
                    vocab_size=len(graph_builder.EdgeType.__members__),
                    embed_dim=self._edge_embedding_size,
                    initializer=embedding_initializer,
                ),
                node_model_fn=functools.partial(
                    TokenGraphBuilderModelNodeEmbed,
                    vocab_size=len(self._token_list),
                    common_embed_dim=self._common_node_embedding_size,
                    num_annotations=self._num_annotations,
                    initializer=embedding_initializer,
                ),
                global_model_fn=functools.partial(
                    snt.Sequential,
                    (
                        model_blocks.CastLayer(self.dtype),
                        snt.nets.MLP(
                            output_sizes=(self._global_embedding_size,),
                            **mlp_initializers,
                            activation=self._update_activation,
                        ),
                    ),
                ),
                name='encoder',
            ),
            num_iterations=1,
            layer_normalization=options.EnableFeature.NEVER,
            residual_connection=options.EnableFeature.NEVER,
            edges_output_size=(None, self._edge_embedding_size),
            nodes_output_size=(None, self._node_embedding_size),
            globals_output_size=(None, self._global_embedding_size),
            extra_node_inputs=(
                'instruction_node_mask',
                'instruction_annotations',
            ),
        ),
        gnn_model_base.GraphNetworkLayer(
            module=graph_nets.modules.GraphNetwork(
                edge_model_fn=functools.partial(
                    snt.nets.MLP,
                    output_sizes=self._edge_update_layers,
                    **mlp_initializers,
                    activation=self._update_activation,
                ),
                node_model_fn=functools.partial(
                    snt.nets.MLP,
                    output_sizes=self._node_update_layers,
                    **mlp_initializers,
                    activation=self._update_activation,
                ),
                node_block_opt={
                    'use_sent_edges': self._use_sent_edges,
                },
                global_model_fn=functools.partial(
                    snt.nets.MLP,
                    output_sizes=self._global_update_layers,
                    **mlp_initializers,
                    activation=self._update_activation,
                ),
                name='network',
            ),
            num_iterations=None,
            layer_normalization=options.EnableFeature.BY_FLAG,
            residual_connection=options.EnableFeature.BY_FLAG,
            edges_output_size=(None, self._edge_update_layers[-1]),
            nodes_output_size=(None, self._node_update_layers[-1]),
            globals_output_size=(None, self._global_update_layers[-1]),
        ),
    )


class TokenGraphBuilderModelNodeEmbed:
  """Class representing node embeddings with instruction annotations included.

  `snt.Embed`-like class. Generates node embeddings normally, then replaces the
  last `num_annotation` values of the embeddings corresponding to instructions
  with the annotation values. The embeddings for other node types remain
  unchanged.
  """

  def __init__(
      self,
      common_embed_dim,
      num_annotations,
      **kwargs,
  ) -> None:
    """Initializes node embeddings.

    Args:
      common_embed_dim: The length of the learnt part of the instruction node
        embedding vectors. The remainder of the vector is filled with
        instruction annotation.
      num_annotations: The number of annotations per instruction.
      kwargs: Additional arguments to be passed to the internal `snt.Embed`s.
    """

    # The first `embed_dim - num_annotations` embedding values for all nodes.
    self._common_embed = snt.Embed(
        embed_dim=common_embed_dim,
        **kwargs,
    )

    # `num_annotations` extra learnt embedding values for non-instruction nodes.
    # Instruction nodes will use instruction annotations instead of these learnt
    # embeddings. This is not required when there are no annotations - in that
    # case, we simply return the common embeddings.
    self._extra_embed = None
    if num_annotations:
      self._extra_embed = snt.Embed(
          embed_dim=num_annotations,
          **kwargs,
      )

  def __call__(
      self,
      inputs,
      instruction_node_mask,
      instruction_annotations,
  ):
    if not self._extra_embed:
      return self._common_embed(inputs)

    common_embeddings = self._common_embed(inputs)
    extra_embeddings = self._extra_embed(inputs)

    return tf.concat(
        [
            common_embeddings,
            tf.tensor_scatter_nd_update(
                extra_embeddings,
                indices=tf.where(instruction_node_mask),
                updates=instruction_annotations,
            ),
        ],
        axis=1,
    )
