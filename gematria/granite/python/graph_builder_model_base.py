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
"""Provides a base class for models using the BasicBlockGraphBuilder class.

This module provides `GraphBuilderModelBase`, a base class that uses the
BasicBlockGraphBuilder class. All models based on this class use the same graph
format of the basic block, but they can use different ways to process the data:
 - they can use different combinations of graph network modules when computing
   the embeddings for the nodes of the graph,
 - they can use different readout networks to extract/post-process information
   from the graph network.
"""

from collections.abc import Sequence
from typing import Any

from gematria.basic_block.python import basic_block
from gematria.granite.python import gnn_model_base
from gematria.granite.python import graph_builder
from gematria.model.python import model_base
from gematria.model.python import oov_token_behavior
from gematria.model.python import token_model
import graph_nets
import numpy as np
import tensorflow.compat.v1 as tf

_OutOfVocabularyTokenBehavior = oov_token_behavior.OutOfVocabularyTokenBehavior


class GraphBuilderModelBase(
    token_model.TokenModel, gnn_model_base.GnnModelBase
):
  """Base class for models usign the BasicBlockGraphBuilder graphs.

  The model integrates the basic block to graph transformation into the Gematria
  framework and provides functionality for extracting per-instruction data out
  of the output graph. The construction of the graph network modules and the
  readout network must be done in child classes.
  """

  # The name of GraphBuilderModelBase.special_tokens_tensor in the TensorFlow
  # graph.
  SPECIAL_TOKENS_TENSOR_NAME = 'GraphBuilderModelBase.special_tokens'

  # The name of the input tensor that receives the instruction node mask.
  INSTRUCTION_NODE_MASK_TENSOR_NAME = (
      'GraphBuilderModelBase.instruction_node_mask'
  )

  # The name of the input tensor that holds the instruction annotations.
  INSTRUCTION_ANNOTATIONS_TENSOR_NAME = (
      'GraphBuilderModelBase.instruction_annotations'
  )

  # The name of the tensor holding ordered annotation names.
  ANNOTATION_NAMES_TENSOR_NAME = 'GraphBuilderModelBase.annotation_names'

  # A Boolean tensor placeholder that receives a mask for instruction nodes. The
  # mask has shape (None,), and it must have the same length as
  # self._graphs_tuple_placeholders.nodes along the first dimension. It contains
  # True at position i if and only if self._graphs_tuple_placeholders.nodes[i]
  # is an instruction node (node_type == NodeType.INSTRUCTION). The mask is used
  # to collect the feature vectors of nodes corresponding to instructions for
  # further processing during readout.
  _instruction_node_mask: tf.Tensor

  # A tensor that contains feature vectors of nodes representing instructions in
  # the order in which they are in the basic block, i.e. in the same order
  # instructions appear in ModelBase._output_tensor_deltas.
  _instruction_features: tf.Tensor

  # The graph builder used to compose the GraphsTuple data structure passed to
  # the TensorFlow computation.
  _batch_graph_builder: graph_builder.BasicBlockGraphBuilder

  # A 1D int tensor that contains indices of special tokens used by the graph
  # builder. See the docstring of self.special_tokens_tensor for more details on
  # the format of the data.
  _special_tokens_tensor: tf.Tensor

  # A 1D byte tensor that contains the list of annotation names in the order of
  # their indices in the graph builder.
  _annotation_names_tensor: tf.Tensor

  # The list of annotation names, in the order of their indices in the model.
  _annotation_names_list: Sequence[str]

  # A 2D float tensor holding instruction annotations.
  _instruction_annotations: tf.Tensor

  def __init__(
      self,
      *,
      tokens: Sequence[str],
      immediate_token: str,
      fp_immediate_token: str,
      address_token: str,
      memory_token: str,
      annotation_names: Sequence[str] = [],
      **kwargs: Any,
  ) -> None:
    """Initializes the model with the given feature factory.

    Args:
      tokens: The list of tokens that may be associated with the nodes of the
        basic block graph (e.g. instruction mnemonics and register names used in
        the canonicalized basic block protos processed by the model). This list
        is passed to the basic block builder which translates the token for each
        node into its index in this list, and the index is then used as the
        feature of the node.
      immediate_token: The token that is associated with immediate value nodes
        in the basic block graph.
      fp_immediate_token: The token that is associated with floating-point
        immediate values in the basic block graph.
      address_token: The token that is associated with address computation nodes
        in the basic block graph.
      memory_token: The token that is associated with memory value nodes in the
        basic block graph.
      annotation_names: The list of names of annotations to be used.
      **kwargs: Additional keyword arguments are passed to the constructor of
        the base class.
    """
    # NOTE(ondrasej): We set the node/edge feature dtypes to int32. They are
    # indices to the token list/edge type; an int32 should be sufficient for all
    # our use cases and fixing the type will make it easier to move the array
    # construction to the C++ code if needed in the future. Similarly for the
    # graph index dtype.
    super().__init__(
        node_feature_shape=(),
        node_feature_dtype=tf.dtypes.int32,
        edge_feature_shape=(),
        edge_feature_dtype=tf.dtypes.int32,
        global_feature_shape=(len(tokens),),
        global_feature_dtype=tf.dtypes.int32,
        graph_index_dtype=tf.dtypes.int32,
        tokens=tokens,
        **kwargs,
    )
    self._instruction_features = None
    self._batch_graph_builder = graph_builder.BasicBlockGraphBuilder(
        node_tokens=self._token_list,
        immediate_token=immediate_token,
        fp_immediate_token=fp_immediate_token,
        address_token=address_token,
        memory_token=memory_token,
        annotation_names=annotation_names,
        out_of_vocabulary_behavior=self._oov_behavior,
    )

    self._special_tokens_tensor = None

    self._annotation_names_list = tuple(
        self._batch_graph_builder.annotation_names
    )
    self._num_annotations = len(self._annotation_names_list)

  @property
  def special_tokens_tensor(self) -> tf.Tensor:
    """Returns the indices of special node tokens.

    The returned tensor contains indices of the special tokens in the list
    encoded in self.token_list_tensor. The indices of the special tokens are
    stored in the following order:
      1. immediate value node token,
      2. floating-point immediate value node token,
      3. address computation node token,
      4. memory value node token,
      5. replacement token used to replace immediate values. This index is set
         to -1 when the model is not trained with replacement tokens.
    """
    return self._special_tokens_tensor

  @property
  def annotation_names_tensor(self) -> tf.Tensor:
    return self._annotation_names_tensor

  # @Override
  @property
  def output_tensor_names(self) -> Sequence[str]:
    return (
        *super().output_tensor_names,
        GraphBuilderModelBase.SPECIAL_TOKENS_TENSOR_NAME,
        GraphBuilderModelBase.ANNOTATION_NAMES_TENSOR_NAME,
    )

  # @Override
  def _create_tf_graph(self) -> None:
    """See base class."""
    super()._create_tf_graph()
    special_tokens = np.array(
        (
            self._batch_graph_builder.immediate_token,
            self._batch_graph_builder.fp_immediate_token,
            self._batch_graph_builder.address_token,
            self._batch_graph_builder.memory_token,
            self._batch_graph_builder.replacement_token,
        ),
        dtype=np.int32,
    )
    self._special_tokens_tensor = tf.constant(
        special_tokens,
        dtype=tf.dtypes.int32,
        name=GraphBuilderModelBase.SPECIAL_TOKENS_TENSOR_NAME,
    )
    annotation_names_array = np.frombuffer(
        b'\0'.join(
            name.encode('utf-8') for name in self._annotation_names_list
        ),
        dtype=np.uint8,
    )
    self._annotation_names_tensor = tf.constant(
        annotation_names_array,
        name=GraphBuilderModelBase.ANNOTATION_NAMES_TENSOR_NAME,
    )

  # @Override
  def _create_graph_network_resources(self) -> None:
    super()._create_graph_network_resources()
    self._instruction_annotations = tf.placeholder(
        dtype=self.dtype,
        shape=(None, len(self._annotation_names_list)),
        name=GraphBuilderModelBase.INSTRUCTION_ANNOTATIONS_TENSOR_NAME,
    )
    self._instruction_node_mask = tf.placeholder(
        dtype=tf.dtypes.bool,
        shape=(None,),
        name=GraphBuilderModelBase.INSTRUCTION_NODE_MASK_TENSOR_NAME,
    )

  # @Override
  def _create_readout_network_resources(self) -> None:
    super()._create_readout_network_resources()
    self._instruction_features = tf.boolean_mask(
        self._graphs_tuple_outputs.nodes, self._instruction_node_mask
    )

  # @Override
  def _start_batch(self) -> None:
    super()._start_batch()
    self._batch_graph_builder.reset()

  # @Override
  def _make_batch_feed_dict(self) -> model_base.FeedDict:
    feed_dict = super()._make_batch_feed_dict()
    feed_dict[self._instruction_node_mask] = np.array(
        self._batch_graph_builder.instruction_node_mask, dtype=bool
    )
    feed_dict[self._instruction_annotations] = (
        self._batch_graph_builder.instruction_annotations
    )
    return feed_dict

  # @Override
  def _make_batch_graphs_tuple(self):
    node_features = np.array(
        self._batch_graph_builder.node_features,
        dtype=self._graph_node_feature_spec.dtype.as_numpy_dtype,
    )
    if self._oov_injection_probability > 0:
      # Each token is replaced with the probability
      # self._oov_injection_probability.
      # TODO(ondrasej): Consider initializing the random number generator using
      # some property of the batch, to ensure that the replacements for each
      # batch are stable.
      injection_mask = (
          np.random.default_rng().random(node_features.shape)
          < self._oov_injection_probability
      )
      node_features[injection_mask] = self._oov_token
    return graph_nets.graphs.GraphsTuple(
        nodes=node_features,
        edges=np.array(
            self._batch_graph_builder.edge_features,
            dtype=self._graph_edge_feature_spec.dtype.as_numpy_dtype,
        ),
        # NOTE(ondrasej): The graph globals are not normalized by the number of
        # nodes in the graph. We could do it here, but we can also do it by
        # introducing a LayerNorm layer in the first graph network module.
        globals=np.array(
            self._batch_graph_builder.global_features,
            dtype=self._graph_global_feature_spec.dtype.as_numpy_dtype,
        ),
        receivers=np.array(
            self._batch_graph_builder.edge_receivers,
            dtype=self._graph_index_dtype.as_numpy_dtype,
        ),
        senders=np.array(
            self._batch_graph_builder.edge_senders,
            dtype=self._graph_index_dtype.as_numpy_dtype,
        ),
        n_node=np.array(
            self._batch_graph_builder.num_nodes_per_block,
            dtype=self._graph_index_dtype.as_numpy_dtype,
        ),
        n_edge=np.array(
            self._batch_graph_builder.num_edges_per_block,
            dtype=self._graph_index_dtype.as_numpy_dtype,
        ),
    )

  # @Override
  def _add_basic_block_to_batch(self, block: basic_block.BasicBlock) -> None:
    basic_block_was_added = self._batch_graph_builder.add_basic_block(block)
    if not basic_block_was_added:
      # TODO(ondrasej): Better handling of blocks that can't be added to the
      # batch. For now, we just let the exception propagate out of the model and
      # let the user handle it.
      raise model_base.AddBasicBlockError(
          f'Basic block could not be added to the batch: {block}'
      )
