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

from absl.testing import parameterized
from gematria.basic_block.python import basic_block
from gematria.basic_block.python import throughput
from gematria.basic_block.python import tokens
from gematria.granite.python import gnn_model_base
from gematria.granite.python import graph_builder
from gematria.granite.python import graph_builder_model_base
from gematria.model.python import model_blocks
from gematria.model.python import oov_token_behavior
from gematria.model.python import options
from gematria.model.python import token_model
from gematria.testing.python import model_test
import graph_nets
import numpy as np
import sonnet as snt
import tensorflow as tf
import tf_keras

_OutOfVocabularyTokenBehavior = oov_token_behavior.OutOfVocabularyTokenBehavior


class TestGraphBuilderModel(graph_builder_model_base.GraphBuilderModelBase):

  def __init__(
      self,
      out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
      **kwargs
  ):
    super().__init__(
        learning_rate=0.02,
        dtype=tf.dtypes.float32,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        out_of_vocabulary_behavior=out_of_vocabulary_behavior,
        **kwargs,
    )

  def _make_model_name(self):
    return 'TestGraphBuilderModel'

  def _create_graph_network_modules(self):
    embedding_initializer = tf_keras.initializers.glorot_normal()
    mlp_initializers = {
        'w_init': tf_keras.initializers.glorot_normal(),
        'b_init': tf_keras.initializers.glorot_normal(),
    }
    return (
        gnn_model_base.GraphNetworkLayer(
            module=graph_nets.modules.GraphIndependent(
                edge_model_fn=functools.partial(
                    snt.Embed,
                    vocab_size=len(graph_builder.EdgeType),
                    embed_dim=1,
                    initializer=embedding_initializer,
                ),
                node_model_fn=functools.partial(
                    snt.Embed,
                    vocab_size=self._batch_graph_builder.num_node_tokens,
                    embed_dim=1,
                    initializer=embedding_initializer,
                ),
                global_model_fn=functools.partial(
                    model_blocks.CastLayer, self.dtype
                ),
            ),
            num_iterations=1,
            layer_normalization=options.EnableFeature.NEVER,
            residual_connection=options.EnableFeature.NEVER,
        ),
        gnn_model_base.GraphNetworkLayer(
            module=graph_nets.modules.GraphNetwork(
                edge_model_fn=functools.partial(
                    snt.nets.MLP,
                    output_sizes=(1,),
                    **mlp_initializers,
                ),
                node_model_fn=functools.partial(
                    snt.nets.MLP,
                    output_sizes=(1,),
                    **mlp_initializers,
                ),
                global_model_fn=functools.partial(
                    snt.nets.MLP,
                    output_sizes=(1,),
                    **mlp_initializers,
                ),
                name='network',
            ),
            num_iterations=None,
            layer_normalization=options.EnableFeature.BY_FLAG,
            residual_connection=options.EnableFeature.BY_FLAG,
        ),
    )

  def _execute_readout_network(self, graph_tuple, feed_dict):
    if self._use_deltas:
      output = tf.boolean_mask(
          graph_tuple.nodes, feed_dict['instruction_node_mask']
      )
    else:
      output = graph_tuple.globals
    return tf.reshape(tensor=output, shape=[-1, 1])


class GraphBuilderModelBaseTest(parameterized.TestCase, model_test.TestCase):

  def setUp(self):
    # NOTE(ondrasej): We use the three tests only in self.test_schedule_batch().
    # The tests that run training use only one basic block to reduce the time
    # required to reach the precision threshold.
    self.num_blocks = 3
    super().setUp()

  def test_schedule_batch(self):
    model = TestGraphBuilderModel(
        tokens=self.tokens, num_message_passing_iterations=1
    )
    model.initialize()
    schedule = model.schedule_batch(self.blocks_with_throughput)
    self.assertEqual(
        schedule['graph_tuple'].globals.shape,
        (3, len(self.tokens)),
    )
    self.assertEqual(schedule['graph_tuple'].nodes.shape, (26,))
    self.assertAllEqual(schedule['graph_tuple'].edges.shape, (30,))

  def test_node_token_names(self):
    model = TestGraphBuilderModel(
        tokens=self.tokens, num_message_passing_iterations=1
    )
    model.initialize()

    # NOTE(ondrasej): The conversion to a tuple (or another collection) is
    # needed - otherwise, assertAll would treat expected_token_names as
    # a scalar array with a single string value.
    expected_token_names = tuple(
        b'\0'.join(token.encode('utf-8') for token in self.tokens)
    )
    token_names = model.token_list_tensor.numpy()
    self.assertAllEqual(expected_token_names, token_names)

    special_tokens = model.special_tokens_tensor.numpy()
    self.assertEqual(special_tokens.shape, (5,))
    self.assertAllLess(special_tokens, len(self.tokens))
    self.assertAllGreaterEqual(special_tokens, -1)

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2num_model(self, loss_type, loss_normalization):
    model = TestGraphBuilderModel(
        tokens=self.tokens,
        loss_type=loss_type,
        use_deltas=False,
        use_delta_loss=False,
        loss_normalization=loss_normalization,
        num_message_passing_iterations=1,
    )
    model.initialize()
    self.check_training_model(
        model, self.blocks_with_throughput[0:1], num_epochs=50
    )

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2seq_model(self, loss_type, loss_normalization):
    model = TestGraphBuilderModel(
        tokens=self.tokens,
        loss_type=loss_type,
        use_deltas=True,
        use_delta_loss=False,
        loss_normalization=loss_normalization,
        num_message_passing_iterations=1,
    )
    model.initialize()
    self.check_training_model(
        model, self.blocks_with_throughput[0:1], num_epochs=50
    )

  def test_validate_basic_block(self):
    model = TestGraphBuilderModel(
        tokens=self.tokens, num_message_passing_iterations=1
    )
    model.initialize()

    for block in self.blocks_with_throughput:
      self.assertTrue(model.validate_basic_block(block.block))
      self.assertTrue(model.validate_basic_block_with_throughput(block))
    # This basic block is invalid - there is no x86-64 instruction `FOOBAR`.
    invalid_block = throughput.BasicBlockWithThroughput(
        block=basic_block.BasicBlock(
            basic_block.InstructionList((
                basic_block.Instruction(mnemonic='FOOBAR'),
            ))
        )
    )
    self.assertFalse(model.validate_basic_block_with_throughput(invalid_block))
    self.assertFalse(model.validate_basic_block(invalid_block.block))

  def test_inject_out_of_vocabulary_tokens_invalid(self):
    with self.assertRaises(ValueError):
      _ = TestGraphBuilderModel(
          tokens=self.tokens,
          num_message_passing_iterations=1,
          # The default behavior is to return an error, which is not compatible
          # with a non-zero injection probability.
          out_of_vocabulary_injection_probability=1.0,
      )

  def test_inject_out_of_vocabulary_tokens(self):
    oov_behavior = _OutOfVocabularyTokenBehavior.replace_with_token(
        tokens.UNKNOWN
    )
    model = TestGraphBuilderModel(
        tokens=self.tokens,
        num_message_passing_iterations=1,
        out_of_vocabulary_behavior=oov_behavior,
        out_of_vocabulary_injection_probability=1.0,
    )
    model.initialize()

    self.assertGreaterEqual(model._oov_token, 0)

    schedule = model.schedule_batch(self.blocks_with_throughput)
    node_features = schedule['graph_tuple'].nodes
    expected_node_features = np.full_like(node_features, model._oov_token)
    self.assertAllEqual(node_features, expected_node_features)

  def test_inject_out_of_vocabulary_estimate(self):
    oov_behavior = _OutOfVocabularyTokenBehavior.replace_with_token(
        tokens.UNKNOWN
    )
    injection_probability = 0.3
    model = TestGraphBuilderModel(
        tokens=self.tokens,
        num_message_passing_iterations=1,
        out_of_vocabulary_behavior=oov_behavior,
        out_of_vocabulary_injection_probability=injection_probability,
    )
    model.initialize()

    # We compute a point estimate of the injection probability from the
    # generated masks and check that true probability of injection is inside the
    # confidence interval of this estimate. The confidence bounds are chosen so
    # that the probability that this test fails is (far) smaller than 10^-10.
    num_trials = 100
    num_ones = 0
    num_all_elements = 0
    for _ in range(num_trials):
      schedule = model.schedule_batch(self.blocks_with_throughput)
      oov_token_mask = schedule['graph_tuple'].nodes == model._oov_token
      num_ones += sum(oov_token_mask)
      num_all_elements += oov_token_mask.size

    injection_probability_estimate = num_ones / num_all_elements
    self.assertBetween(injection_probability_estimate, 0.2, 0.4)

  def test_inject_out_of_vocabulary_tokens_zero_probability(self):
    oov_behavior = _OutOfVocabularyTokenBehavior.replace_with_token(
        tokens.UNKNOWN
    )
    model = TestGraphBuilderModel(
        tokens=self.tokens,
        num_message_passing_iterations=1,
        out_of_vocabulary_behavior=oov_behavior,
        out_of_vocabulary_injection_probability=0.0,
    )
    model.initialize()

    num_trials = 100

    for _ in range(num_trials):
      # Since the replacement probability is zero, and the test data set does
      # not contain unknown tokens, we expect that the out-of-vocabulary
      # replacement token is never used.
      schedule = model.schedule_batch(self.blocks_with_throughput)
      oov_token_mask = schedule['graph_tuple'].nodes == model._oov_token
      expected_oov_token_mask = np.zeros_like(oov_token_mask)
      self.assertAllEqual(oov_token_mask, expected_oov_token_mask)


if __name__ == '__main__':
  tf.test.main()
