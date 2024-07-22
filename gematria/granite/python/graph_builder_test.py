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

import itertools

from absl.testing import absltest
from gematria.basic_block.python import tokens
from gematria.granite.python import graph_builder
from gematria.model.python import oov_token_behavior
from gematria.testing.python import basic_blocks_with_throughput

# A list of tokens that contains all the "helper" tokens used by the graph
# builder but no tokens for the actual assembly code. Transforming a non-empty
# basic block with a graph builder that uses only these tokens is guaranteed to
# trigger the out-of-vocabulary behavior.
_STRUCTURAL_TOKENS = (
    tokens.ADDRESS,
    tokens.IMMEDIATE,
    tokens.MEMORY,
    tokens.UNKNOWN,
)

_OutOfVocabularyTokenBehavior = oov_token_behavior.OutOfVocabularyTokenBehavior


class BasicBlockGraphBuilderTest(
    basic_blocks_with_throughput.TestCase, absltest.TestCase
):
  """Test for the BasicBlockGraphBuilder class wrapper.

  Most of the functionality is tested in the corresponding cc_test(). Here we
  test just that all methods return data in the expected shape.
  """

  def setUp(self):
    self.num_blocks = 10
    super().setUp()

  def assertBuilderIsSelfConsistent(self, builder, num_blocks):
    self.assertLen(self.tokens, builder.num_node_tokens)

    edge_senders = builder.edge_senders
    edge_receivers = builder.edge_receivers
    self.assertLen(builder.node_features, builder.num_nodes)
    self.assertLen(edge_senders, builder.num_edges)
    self.assertLen(edge_receivers, builder.num_edges)
    self.assertLen(builder.edge_features, builder.num_edges)

    self.assertLen(builder.instruction_node_mask, builder.num_nodes)

    num_nodes_per_block = builder.num_nodes_per_block
    num_edges_per_block = builder.num_edges_per_block
    self.assertLen(num_nodes_per_block, num_blocks)
    self.assertLen(num_edges_per_block, num_blocks)

    self.assertEqual(builder.num_nodes, sum(num_nodes_per_block))
    self.assertEqual(builder.num_edges, sum(num_edges_per_block))

    global_features = builder.global_features
    self.assertLen(global_features, num_blocks)
    for global_feature in global_features:
      self.assertLen(global_feature, builder.num_node_tokens)

    if builder.num_edges:
      self.assertLess(max(edge_senders), builder.num_edges)
      self.assertLess(max(edge_receivers), builder.num_nodes)
      self.assertGreaterEqual(min(edge_senders), 0)
      self.assertGreaterEqual(min(edge_receivers), 0)

    num_instruction_nodes = sum(builder.instruction_node_mask)
    num_annotation_types = len(builder.annotation_names)
    self.assertLen(builder.instruction_annotations, num_instruction_nodes)
    for row in builder.instruction_annotations:
      self.assertLen(row, num_annotation_types)

  def test_single_instruction_basic_block(self):
    builder = graph_builder.BasicBlockGraphBuilder(
        node_tokens=self.tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        annotation_names=set(),
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
    )

    self.assertEqual(builder.num_nodes, 0)
    self.assertEqual(builder.num_edges, 0)
    self.assertBuilderIsSelfConsistent(builder, 0)

    self.assertTrue(builder.add_basic_block(self.blocks[0]))

    self.assertEqual(builder.num_graphs, 1)
    self.assertGreater(builder.num_nodes, 0)
    self.assertGreater(builder.num_edges, 0)
    self.assertBuilderIsSelfConsistent(builder, 1)

    builder.reset()

    self.assertEqual(builder.num_graphs, 0)
    self.assertEqual(builder.num_nodes, 0)
    self.assertEqual(builder.num_edges, 0)
    self.assertBuilderIsSelfConsistent(builder, 0)

  def test_multiple_basic_blocks(self):
    builder = graph_builder.BasicBlockGraphBuilder(
        node_tokens=self.tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        annotation_names=set(),
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
    )

    self.assertTrue(builder.add_basic_block(self.blocks[0]))
    self.assertTrue(builder.add_basic_block(self.blocks[1]))

    self.assertBuilderIsSelfConsistent(builder, 2)

  def test_multiple_annotated_basic_blocks(self):
    builder = graph_builder.BasicBlockGraphBuilder(
        node_tokens=self.tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        annotation_names=self.annotation_names,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
    )

    self.assertTrue(builder.add_basic_block(self.annotated_blocks[0]))
    self.assertTrue(builder.add_basic_block(self.annotated_blocks[1]))
    self.assertTrue(builder.add_basic_block(self.annotated_blocks[2]))
    self.assertTrue(builder.add_basic_block(self.annotated_blocks[3]))

    self.assertBuilderIsSelfConsistent(builder, 4)

  def test_many_blocks(self):
    builder = graph_builder.BasicBlockGraphBuilder(
        node_tokens=self.tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        annotation_names=set(),
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
    )

    num_blocks = 1000
    for block in itertools.islice(itertools.cycle(self.blocks), num_blocks):
      self.assertTrue(builder.add_basic_block(block))

    self.assertBuilderIsSelfConsistent(builder, num_blocks)

  def test_out_of_vocabulary_tokens_return_error(self):
    builder = graph_builder.BasicBlockGraphBuilder(
        node_tokens=_STRUCTURAL_TOKENS,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        annotation_names=set(),
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
    )

    for block in self.blocks:
      self.assertFalse(builder.add_basic_block(block))

    # Check that no blocks were added to the graph builder.
    self.assertEqual(builder.num_graphs, 0)
    self.assertEqual(builder.num_nodes, 0)
    self.assertEqual(builder.num_edges, 0)

  def test_out_of_vocabulary_tokens_replace_token(self):
    builder = graph_builder.BasicBlockGraphBuilder(
        node_tokens=_STRUCTURAL_TOKENS,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        annotation_names=set(),
        out_of_vocabulary_behavior=(
            _OutOfVocabularyTokenBehavior.replace_with_token(tokens.UNKNOWN)
        ),
    )

    for block in self.blocks:
      self.assertTrue(builder.add_basic_block(block))

    self.assertLen(self.blocks, builder.num_graphs)


if __name__ == '__main__':
  absltest.main()
