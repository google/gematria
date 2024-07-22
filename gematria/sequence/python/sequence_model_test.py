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

from absl.testing import parameterized
from gematria.basic_block.python import basic_block
from gematria.basic_block.python import tokens
from gematria.model.python import oov_token_behavior
from gematria.model.python import token_model
from gematria.sequence.python import sequence_model
from gematria.testing.python import model_test
import numpy as np
import tensorflow.compat.v1 as tf
import tf_keras

_OutOfVocabularyTokenBehavior = oov_token_behavior.OutOfVocabularyTokenBehavior


class TestSequenceModel(sequence_model.SequenceModelBase):
  """A simple sequence-based model used in the tests.

  The model has an embedding vector (with one element per task) for each token,
  and computes the prediction as the sum of all token embedding vectors in the
  basic block.
  """

  def __init__(
      self,
      out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
      **kwargs
  ):
    super().__init__(
        learning_rate=0.02,
        dtype=tf.dtypes.float32,
        out_of_vocabulary_behavior=out_of_vocabulary_behavior,
        **kwargs
    )

  def _make_model_name(self):
    return 'TestSequenceModel'

  def _create_model(self):
    token_sequence = tf_keras.Input(shape=(), dtype=tf.dtypes.int32)
    num_tokens_per_instruction = tf_keras.Input(shape=(), dtype=tf.dtypes.int32)
    num_instructions_per_block = tf_keras.Input(shape=(), dtype=tf.dtypes.int32)

    embed_input = tf_keras.layers.Embedding(
        input_dim=len(self._token_list), output_dim=self.num_tasks
    )
    embedded_tokens = embed_input(token_sequence)
    instructions = tf.RaggedTensor.from_row_lengths(
        embedded_tokens, num_tokens_per_instruction
    )

    instruction_values = tf.math.reduce_sum(instructions, axis=1)
    if self.use_deltas:
      output = instruction_values
    else:
      block_tensor = tf.RaggedTensor.from_row_lengths(
          instruction_values, num_instructions_per_block
      )
      output = tf.math.reduce_sum(block_tensor, axis=1)

    output = tf.reshape(output, (-1, self.num_tasks))
    return tf_keras.Model(
        inputs=(
            token_sequence,
            num_tokens_per_instruction,
            num_instructions_per_block,
        ),
        outputs=output,
    )


class SequenceModelTest(parameterized.TestCase, model_test.TestCase):

  def setUp(self):
    # NOTE(ondrasej): We use the three tests only in self.test_schedule_batch().
    # The tests that run training use only one basic block to reduce the time
    # required to reach the precision threshold.
    self.num_blocks = 3
    super().setUp()

  @parameterized.named_parameters(('seq2num', False), ('seq2seq', True))
  def test_schedule_batch(self, use_deltas):
    model = TestSequenceModel(tokens=self.tokens, use_deltas=use_deltas)
    model.initialize()
    schedule = model.schedule_batch(self.blocks_with_throughput)
    self.assertEqual(schedule[model._token_sequence_placeholder].shape, (58,))
    self.assertEqual(
        schedule[model._num_tokens_per_instruction_placeholder].shape, (7,)
    )
    self.assertEqual(
        schedule[model._num_instructions_per_block_placeholder].shape, (3,)
    )

  def test_schedule_batch_with_invalid_block(self):
    model = TestSequenceModel(tokens=self.tokens)
    model.initialize()

    invalid_block = basic_block.BasicBlock(
        instructions=basic_block.InstructionList((
            basic_block.Instruction(mnemonic='FOOBAR'),
        ))
    )
    self.assertFalse(model.validate_basic_block(invalid_block))

    with self.assertRaises(token_model.TokenNotFoundError):
      model.schedule_batch((invalid_block,))

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2seq(self, loss_type, loss_normalization):
    model = TestSequenceModel(
        tokens=self.tokens,
        use_deltas=True,
        loss_type=loss_type,
        loss_normalization=loss_normalization,
    )
    model.initialize()
    self.check_training_model(model, self.blocks_with_throughput[0:1])

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2num(self, loss_type, loss_normalization):
    model = TestSequenceModel(
        tokens=self.tokens,
        use_deltas=False,
        loss_type=loss_type,
        loss_normalization=loss_normalization,
    )
    model.initialize()
    self.check_training_model(model, self.blocks_with_throughput[0:1])

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2num_multi_task(self, loss_type, loss_normalization):
    model = TestSequenceModel(
        tokens=self.tokens,
        use_deltas=False,
        task_list=('llvm', 'test'),
        loss_type=loss_type,
        loss_normalization=loss_normalization,
    )
    model.initialize()
    self.check_training_model(model, self.blocks_with_throughput[0:1])

  def test_inject_out_of_vocabulary_tokens_invalid(self):
    with self.assertRaises(ValueError):
      _ = TestSequenceModel(
          tokens=self.tokens,
          # The default behavior is to return an error, which is not compatible
          # with a non-zero injection probability.
          out_of_vocabulary_injection_probability=1.0,
      )

  def test_inject_out_of_vocabulary_tokens(self):
    oov_behavior = _OutOfVocabularyTokenBehavior.replace_with_token(
        tokens.UNKNOWN
    )
    model = TestSequenceModel(
        tokens=self.tokens,
        out_of_vocabulary_behavior=oov_behavior,
        out_of_vocabulary_injection_probability=1.0,
    )
    model.initialize()

    self.assertGreaterEqual(model._oov_token, 0)

    schedule = model.schedule_batch(self.blocks_with_throughput)
    token_sequence = schedule[model._token_sequence_placeholder]
    expected_token_sequence = np.full_like(token_sequence, model._oov_token)
    self.assertAllEqual(token_sequence, expected_token_sequence)

  def test_inject_out_of_vocabulary_estimate(self):
    oov_behavior = _OutOfVocabularyTokenBehavior.replace_with_token(
        tokens.UNKNOWN
    )
    injection_probability = 0.3
    model = TestSequenceModel(
        tokens=self.tokens,
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
      oov_token_mask = (
          schedule[model._token_sequence_placeholder] == model._oov_token
      )
      num_ones += sum(oov_token_mask)
      num_all_elements += oov_token_mask.size

    injection_probability_estimate = num_ones / num_all_elements
    self.assertBetween(injection_probability_estimate, 0.2, 0.4)

  def test_inject_out_of_vocabulary_tokens_zero_probability(self):
    oov_behavior = _OutOfVocabularyTokenBehavior.replace_with_token(
        tokens.UNKNOWN
    )
    model = TestSequenceModel(
        tokens=self.tokens,
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
      oov_token_mask = (
          schedule[model._token_sequence_placeholder] == model._oov_token
      )
      expected_oov_token_mask = np.zeros_like(oov_token_mask)
      self.assertAllEqual(oov_token_mask, expected_oov_token_mask)

  def test_validate_basic_block(self):
    model = TestSequenceModel(
        tokens=self.tokens,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
    )
    model.initialize()

    for block in self.blocks_with_throughput:
      self.assertTrue(model.validate_basic_block(block.block))
      self.assertTrue(model.validate_basic_block_with_throughput(block))

    invalid_block = basic_block.BasicBlock(
        instructions=basic_block.InstructionList((
            basic_block.Instruction(mnemonic='FOOBAR'),
        ))
    )
    self.assertFalse(model.validate_basic_block(invalid_block))


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
