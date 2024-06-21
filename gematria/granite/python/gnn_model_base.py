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
"""Provides a base class for models based on DeepMind's graph_nets library.

See the documentation of GnnModelBase for more details.
"""

import abc
from collections.abc import Sequence
import dataclasses
from typing import Optional

from gematria.model.python import model_base
from gematria.model.python import model_blocks
from gematria.model.python import options
import graph_nets
import sonnet as snt
import tensorflow.compat.v1 as tf


def _add_batch_dimension(shape: Sequence[int]) -> Sequence[Optional[int]]:
  """Adds a batch dimension as the first dimension to a given shape.

  Args:
    shape: The shape to which the dimension is added. The size in all dimensions
      must be at least one. Empty shape (i.e. the shape of scalar values) is
      allowed and it will produce a 1D tensor.

  Returns:
    The shape with the batch dimension.

  Raises:
    ValueError: When the input shape is not valid.
  """
  if any(size <= 0 for size in shape):
    raise ValueError('The shape may contain only positive numbers')
  return (None, *shape)


@dataclasses.dataclass(frozen=True)
class GraphNetworkLayer:
  """Specifies one segment of the pipeline of the graph network.

  Each segment consists of a graph network module, i.e. a Sonnet module that
  takes and returns a GraphsTuple and produces a GraphsTuple. Optionally, the
  layer can specify a fixed number of message passing iterations (repetitions)
  of this module. When the number of iterations is not specified, the parameter
  num_message_passing_iterations from the model is used.

  Note that the shapes of inputs and outputs of modules with more than one
  iteration must be compatible.

  Attributes:
    module: A Sonnet module that takes a graph_nets.graphs.GraphsTuple as input
      and produces a graph_nets.graphs.GraphsTuple as output. Typically, this
      would be one of the modules defined in graphs_net.modules.
    num_iterations: The number of repetitions of the module in the graph neural
      network computation. When None, the module is used
      GnnModelBase._num_message_passing_iterations times, according to the model
      it is used in.
    layer_normalization: Determines whether layer normalization is used after
      the graph network layer. When layer_normalization is
      EnableFeature.BY_FLAG, the layer normalization is used when model is
      created with graph_module_layer_normalization=True.
    residual_connection: Determines whether a residual connection is used for
      the graph layer. When a residual connection is used, the input node, edge,
      and global features are added to the output node, edge, and global
      features computed by the layer. When both a residual connection and layer
      normalization are used, the layer normalization op is inserted after the
      residual connection.
  """

  # NOTE(ondrasej): This should be one of the classes defined in
  # graph_nets.modules, but their common base class is private in the library.
  # Since that private class inherits from sonnet.Module, that's what we use
  # here.
  module: snt.Module
  num_iterations: Optional[int]
  layer_normalization: options.EnableFeature
  residual_connection: options.EnableFeature


class GnnModelBase(model_base.ModelBase):
  """Base class for models built using the graph_nets library.

  Each model based on this class has two parts:
   * The Graph neural net that computes the embedding vectors for objects in
     the basic block.
   * A readout network that computes the output values based on the embedding
     vectors.
  When processing a basic block, the model first translates it into graph. This
  graph is presented to a graph neural network that computes embedding vectors
  for nodes, edges and the block itself. The embedding vectors after the last
  iteration of message passing are then given to the readout network that
  produces the outputs of the model.

  The way how a basic block is encoded into the graph, the graph neural network
  used to process it and the readout network are all model-specific, and they
  can be configured by overriding methods of this class.

  When implementing a model, the user must override at least the following
  methods:
   * GnnModelBase._create_graph_networkModel() to create the graph network
     module.
   * GnnModelBase._create_readout_network() to create the readout network based
     on the outputs from the graph network.
   * ModelBase._start_batch(), ModelBase._add_basic_block_to_batch(),
     ModelBase._make_batch_feed_dict(), and/or ModelBase._finalize_batch() to
     provide the code that transforms blocks from the batch to a graph.

  See TestGnnModel and TestEncoderDecoderGnnModel in gnn_model_base_test.py for
  example models built with this class.
  """

  # The names of the tensors that are part of the "public interface" of the
  # model, i.e. placeholders that receive inputs or tensors that contain the
  # outputs of the model.
  NODES_TENSOR_NAME = 'GnnModelBase.node_features'
  EDGES_TENSOR_NAME = 'GnnModelBase.edge_features'
  GLOBALS_TENSOR_NAME = 'GnnModelBase.global_features'
  SENDERS_TENSOR_NAME = 'GnnModelBase.senders'
  RECEIVERS_TENSOR_NAME = 'GnnModelBase.receivers'
  NUM_NODES_TENSOR_NAME = 'GnnModelBase.num_nodes'
  NUM_EDGES_TENSOR_NAME = 'GnnModelBase.num_edges'

  # The DType of the indices of the graph.
  _graph_index_dtype: tf.dtypes.DType

  # The shapes and dtypes of edge, node, and global feature vectors given to the
  # first layer of the graph neural network. This is used in the model in the
  # following ways:
  #  - The shape and the dtype of the placeholders in
  #    self._graphs_tuple_placeholders is based on these values.
  #  - The feature vectors added to the networkx graph objects when scheduling a
  #    batch must have these shapes and dtypes.
  _graph_node_feature_spec: tf.TensorSpec
  _graph_edge_feature_spec: tf.TensorSpec
  _graph_global_feature_spec: tf.TensorSpec

  # The specification of computations on the graph representation. The modules
  # are applied in the order in which they appear in the sequence.
  _graph_network: Sequence[GraphNetworkLayer]
  # A GraphsTuple object with placeholders, used to feed input data to the graph
  # neural net.
  _graphs_tuple_placeholders: graph_nets.graphs.GraphsTuple
  # A GraphsTuple that contains tensors with outputs from the last iteration of
  # message passing in the graph neural net.
  _graphs_tuple_outputs: graph_nets.graphs.GraphsTuple

  # Controls whether layer normalization is applied to node, edge, and feature
  # vectors of graph network layers where layer_normalization is
  # EnableFeature.BY_FLAG.
  _graph_module_layer_normalization: bool

  # The number of message passing iterations done when computing the embedding
  # vectors with the graph neural net.
  _num_message_passing_iterations: int

  def __init__(
      self,
      *,
      node_feature_shape: Sequence[int],
      edge_feature_shape: Sequence[int],
      global_feature_shape: Sequence[int],
      num_message_passing_iterations: int,
      graph_module_residual_connections: bool = False,
      graph_module_layer_normalization: bool = True,
      graph_index_dtype: tf.dtypes.DType = tf.dtypes.int32,
      node_feature_dtype: Optional[tf.dtypes.DType] = None,
      edge_feature_dtype: Optional[tf.dtypes.DType] = None,
      global_feature_dtype: Optional[tf.dtypes.DType] = None,
      **kwargs,
  ) -> None:
    """Creates a new instance of the model with the given parameters.

    Args:
      node_feature_shape: The shape of node feature vectors used as the input of
        the graph neural network.
      edge_feature_shape: The shape of edge feature vectors used as the input of
        the graph neural network.
      global_feature_shape: The shape of global (graph) feature vectors used as
        the input of the neural network.
      num_message_passing_iterations: The number of message passing iterations
        done when computing the embedding vectors with the graph neural net.
      graph_module_residual_connections: When True, residual connections are
        added around the update functions of graph modules where
        residual_connection is EnableFeature.BY_FLAG.
      graph_module_layer_normalization: When True, layer normalization is
        applied to the outputs of each update function of graph network layers
        where layer_normalization is EnableFeature.BY_FLAG.
      graph_index_dtype: The DType used in tensors that contain indices of nodes
        and edges in the input graph.
      node_feature_dtype: The dtype of the node feature vectors used as the
        input of the graph neural network. If unspecified, uses the model dtype.
      edge_feature_dtype: The dtype of the edge feature vectors used as the
        input of the graph neural network. If unspecified, uses the model dtype.
      global_feature_dtype: The dtype of the global feature vectors used as the
        input of the graph neural network. If unspecified, uses the model dtype.
      **kwargs: Additional keyword arguments are passed to ModelBase.__init__().
    """
    super().__init__(**kwargs)
    self._graph_index_dtype = graph_index_dtype
    self._graph_node_feature_spec = tf.TensorSpec(
        shape=node_feature_shape, dtype=node_feature_dtype or self.dtype
    )
    self._graph_edge_feature_spec = tf.TensorSpec(
        shape=edge_feature_shape, dtype=edge_feature_dtype or self.dtype
    )
    self._graph_global_feature_spec = tf.TensorSpec(
        shape=global_feature_shape, dtype=global_feature_dtype or self.dtype
    )

    self._graph_network = None
    self._graphs_tuple_placeholders = None
    self._graphs_tuple_outputs = None

    self._num_message_passing_iterations = num_message_passing_iterations
    self._graph_module_residual_connections = graph_module_residual_connections
    self._graph_module_layer_normalization = graph_module_layer_normalization

  # @Override
  def _create_tf_graph(self) -> None:
    self._create_graph_network_resources()
    self._graph_network = self._create_graph_network_modules()
    assert self._graph_network is not None
    self._graphs_tuple_placeholders = self._create_graphs_placeholders()
    assert self._graphs_tuple_placeholders is not None
    self._graphs_tuple_outputs = self._create_graph_network()
    assert self._graphs_tuple_outputs is not None
    self._create_readout_network_resources()
    if not self._use_deltas:
      self._output_tensor = self._create_readout_network()
    else:
      self._output_tensor_deltas = self._create_readout_network()

  @abc.abstractmethod
  def _create_graph_network_modules(self) -> Sequence[GraphNetworkLayer]:
    """Creates the graph network modules used to compute graph embeddings.

    The returned object defines a sequence of Sonnet modules, where each module
    consumes and produces a GraphsTuple, optionally with a number of iterations.
    The first module consumes the GraphsTuple from
    self._graphs_tuple_placeholders; each following module consumes the output
    of the previous module.

    In a typical use case, there will be two or three modules:
      1. An encoder that takes the feature vectors from the input basic block
         graphs, e.g. graph_nets.modules.GraphIndependent.
      2. The "main" graph neural network, used multiple times.
      3. A decoder that transforms the features computed using message passing
         to the final output format.

    See the class TestEncoderDecoderGnnModel in
    gematria/granite/python/gnn_model_base_test.py
    for an example of a model that uses both an encoder and a decoder module.

    Returns:
      A sequence of GraphNetworkLayer objects that describes the computation in
      the graph network.
    """
    # TODO(ondrasej): Consider specifying the graph network module as a
    # parameter of the constructor instead of filling it in via inheritance.
    raise NotImplementedError(
        'GnnModelBase._create_graph_networkModule is abstract'
    )

  def _create_graphs_placeholders(self) -> graph_nets.graphs.GraphsTuple:
    """Creates placeholder inputs for the graph neural network.

    Returns:
      A GraphsTuple object in which all members are tf.placeholder nodes and
      that is filled in through a feed_dict created during graph scheduling.
    """
    return graph_nets.graphs.GraphsTuple(
        nodes=tf.placeholder(
            dtype=self._graph_node_feature_spec.dtype,
            shape=_add_batch_dimension(self._graph_node_feature_spec.shape),
            name=GnnModelBase.NODES_TENSOR_NAME,
        ),
        edges=tf.placeholder(
            dtype=self._graph_edge_feature_spec.dtype,
            shape=_add_batch_dimension(self._graph_edge_feature_spec.shape),
            name=GnnModelBase.EDGES_TENSOR_NAME,
        ),
        globals=tf.placeholder(
            dtype=self._graph_global_feature_spec.dtype,
            shape=_add_batch_dimension(self._graph_global_feature_spec.shape),
            name=GnnModelBase.GLOBALS_TENSOR_NAME,
        ),
        receivers=tf.placeholder(
            dtype=self._graph_index_dtype,
            shape=(None,),
            name=GnnModelBase.RECEIVERS_TENSOR_NAME,
        ),
        senders=tf.placeholder(
            dtype=self._graph_index_dtype,
            shape=(None,),
            name=GnnModelBase.SENDERS_TENSOR_NAME,
        ),
        n_node=tf.placeholder(
            dtype=self._graph_index_dtype,
            shape=(None,),
            name=GnnModelBase.NUM_NODES_TENSOR_NAME,
        ),
        n_edge=tf.placeholder(
            dtype=self._graph_index_dtype,
            shape=(None,),
            name=GnnModelBase.NUM_EDGES_TENSOR_NAME,
        ),
    )

  def _create_graph_network_resources(self) -> None:
    """Creates resources (like TensorFlow ops) needed by the readout network.

    Child classes can override this method to create resources (e.g. TensorFlow
    ops) that will be needed during the creation of the graph network, but
    that are impractical to create in self._create_graph_network_modules(), e.g.
    to make it easy to override in child classes.

    This method is called before self._create_graph_network_modules().

    By default, this method is a no-op.
    """

  def _create_graph_network(self) -> graph_nets.graphs.GraphsTuple:
    """Creates TensorFlow ops for the graph network.

    By default, this is done by applying the graph network module on the input
    GraphsTuple self._num_message_passing_iterations times.

    Returns:
      The GraphsTuple that contains the outputs of the last application of the
      graph network module.
    """
    graphs_tuple = self._graphs_tuple_placeholders
    for layer_index, layer in enumerate(self._graph_network):
      num_iterations = (
          layer.num_iterations or self._num_message_passing_iterations
      )
      use_residual_connections = (
          layer.residual_connection == options.EnableFeature.ALWAYS
          or (
              layer.residual_connection == options.EnableFeature.BY_FLAG
              and self._graph_module_residual_connections
          )
      )
      use_layer_norm = (
          layer.layer_normalization == options.EnableFeature.ALWAYS
          or (
              layer.layer_normalization == options.EnableFeature.BY_FLAG
              and self._graph_module_layer_normalization
          )
      )
      for iteration in range(num_iterations):
        residual_input = graphs_tuple
        graphs_tuple = layer.module(graphs_tuple)
        if use_residual_connections:
          residual_op_name_base = (
              f'residual_connection_{layer_index}_{iteration}'
          )
          graphs_tuple = graph_nets.graphs.GraphsTuple(
              nodes=model_blocks.add_residual_connection(
                  output_part=graphs_tuple.nodes,
                  residual_part=residual_input.nodes,
                  name=residual_op_name_base + '_nodes',
              ),
              edges=model_blocks.add_residual_connection(
                  output_part=graphs_tuple.edges,
                  residual_part=residual_input.edges,
                  name=residual_op_name_base + '_edges',
              ),
              globals=model_blocks.add_residual_connection(
                  output_part=graphs_tuple.globals,
                  residual_part=residual_input.globals,
                  name=residual_op_name_base + '_globals',
              ),
              receivers=graphs_tuple.receivers,
              senders=graphs_tuple.senders,
              n_node=graphs_tuple.n_node,
              n_edge=graphs_tuple.n_edge,
          )
        if use_layer_norm:
          # Create a new layer normalization step (with a separate scaling
          # factor and bias) per graph network module iteration.
          layer_norm_name_base = (
              f'graph_network_layer_norm_{layer_index}_{iteration}'
          )
          nodes_layer_norm = tf.keras.layers.LayerNormalization(
              name=layer_norm_name_base + '_nodes'
          )
          edges_layer_norm = tf.keras.layers.LayerNormalization(
              name=layer_norm_name_base + '_edges'
          )
          globals_layer_norm = tf.keras.layers.LayerNormalization(
              name=layer_norm_name_base + '_globals'
          )
          graphs_tuple = graph_nets.graphs.GraphsTuple(
              nodes=nodes_layer_norm(graphs_tuple.nodes),
              edges=edges_layer_norm(graphs_tuple.edges),
              globals=globals_layer_norm(graphs_tuple.globals),
              receivers=graphs_tuple.receivers,
              senders=graphs_tuple.senders,
              n_node=graphs_tuple.n_node,
              n_edge=graphs_tuple.n_edge,
          )
    return graphs_tuple

  def _create_readout_network_resources(self) -> None:
    """Creates resources (like TensorFlow ops) needed by the readout network.

    Child classes can override this method to create resources (e.g. TensorFlow
    ops) that will be needed during the creation of the readout network, but
    that are impractical to create in self._create_readout_network(), e.g. to
    make it easy to override in child classes.

    This method is called after self._create_graph_network(), i.e.
    self._graphs_tuple_output is already available, but before the call to
    self._create_readout_network().

    By default, this method is a no-op.
    """

  @abc.abstractmethod
  def _create_readout_network(self) -> tf.Tensor:
    """Creates a readout part of the network.

    Creates TensorFlow ops that take the output of the graph network and
    transform it into a final output.

    Returns:
      A tensor that contains the output of the network. When self._use_deltas is
      True, this output is used as self._output_tensor_deltas; otherwise, it is
      used as self._output_tensor.
    """
    raise NotImplementedError(
        'GnnModelBase._create_readout_network is abstract'
    )

  # @Override
  def _make_batch_feed_dict(self) -> model_base.FeedDict:
    graphs_tuple = self._make_batch_graphs_tuple()
    return graph_nets.utils_tf.get_feed_dict(
        self._graphs_tuple_placeholders, graphs_tuple
    )

  @abc.abstractmethod
  def _make_batch_graphs_tuple(self) -> graph_nets.graphs.GraphsTuple:
    """Creates a graph_nets.GraphsTuple for basic blocks in the current batch.

    This method is called by self._make_batch_feed_dict() as a part of the
    creation of the feed_dict for the current batch. Child classes must override
    it to provide the model-specific graph representation of the basic blocks.
    """
    raise NotImplementedError(
        'GnnModelBase._make_batch_graphs_tuple is abstract'
    )
