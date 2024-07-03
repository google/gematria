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

from gematria.basic_block.python import basic_block
from gematria.basic_block.python import throughput
from gematria.basic_block.python import tokens
from gematria.model.python import oov_token_behavior
from gematria.model.python import token_model
from gematria.testing.python import model_test
import tensorflow.compat.v1 as tf

_OutOfVocabularyTokenBehavior = oov_token_behavior.OutOfVocabularyTokenBehavior


class TestTokenModel(token_model.TokenModel):
  """Minimal working class based on TokenModel.

  This is the minimal class based on TokenModel that can be instantiated, not a
  complete and functional model; it is meant only for testing methods introduced
  in TokenModel, not for testing the whole training/inference pipeline.
  """

  def _create_tf_graph(self):
    super()._create_tf_graph()
    self._output_tensor = tf.placeholder(
        dtype=self.dtype, shape=(None, self.num_tasks)
    )

  def _create_optimizer(self):
    # We can't create an optimizer: this model doesn't have any variables that
    # could be optimized, and the TF optimizers raise an exception in this
    # situation. This is not a problem - this test module does not test training
    # anyway.
    self._decayed_learning_rate = 0.001
    pass

  def _make_model_name(self):
    return 'TestTokenModel'

  def _make_batch_feed_dict(self):
    raise NotImplementedError()

  def _add_basic_block_to_batch(self, block):
    raise NotImplementedError()


class TokenModelTest(model_test.TestCase):

  def test_token_list_tensor(self):
    model = TestTokenModel(
        dtype=tf.dtypes.float32,
        tokens=self.tokens,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
    )
    model.initialize()

    self.assertSequenceEqual(model._token_list, self.tokens)
    with self.session() as sess:
      raw_token_list = sess.run(model.token_list_tensor)
    token_list = raw_token_list.tobytes().split(b'\0')
    self.assertLen(token_list, len(set(token_list)))
    for token in self.tokens:
      self.assertIn(token.encode(), token_list)
    for token in token_list:
      self.assertIn(token.decode(), self.tokens)

  def test_token_index_return_error(self):
    model = TestTokenModel(
        dtype=tf.dtypes.float32,
        tokens=self.tokens,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
    )
    model.initialize()

    for token in self.tokens:
      self.assertGreaterEqual(model._token_index(token), 0)
    with self.assertRaises(token_model.TokenNotFoundError):
      model._token_index('FOOBAR')

  def test_token_index_replace_with_token(self):
    model = TestTokenModel(
        dtype=tf.dtypes.float32,
        tokens=self.tokens,
        out_of_vocabulary_behavior=(
            _OutOfVocabularyTokenBehavior.replace_with_token(tokens.UNKNOWN)
        ),
    )
    model.initialize()

    for token in self.tokens:
      token_index = model._token_index(token)
      self.assertGreaterEqual(token_index, 0)
      self.assertEqual(token_index, self.tokens.index(token))

    self.assertEqual(
        model._token_index('FOOBAR'), self.tokens.index(tokens.UNKNOWN)
    )

  def test_validate_basic_block(self):
    model = TestTokenModel(
        dtype=tf.dtypes.float32,
        tokens=self.tokens,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
    )
    model.initialize()

    for block in self.blocks_with_throughput:
      self.assertTrue(model.validate_basic_block(block.block))
      self.assertTrue(model.validate_basic_blockTokens(block.block))
      self.assertTrue(model.validate_basic_block_with_throughput(block))

    invalid_block = basic_block.BasicBlock(
        basic_block.InstructionList(
            (basic_block.Instruction(mnemonic='FOOBAR'),)
        )
    )
    self.assertFalse(model.validate_basic_block(invalid_block))
    self.assertFalse(model.validate_basic_blockTokens(invalid_block))
    invalid_block_with_throughput = throughput.BasicBlockWithThroughput(
        block=invalid_block, throughputs=()
    )
    self.assertFalse(
        model.validate_basic_block_with_throughput(
            invalid_block_with_throughput
        )
    )

  def test_validate_basic_block_with_replacement_token(self):
    model = TestTokenModel(
        dtype=tf.dtypes.float32,
        tokens=self.tokens,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.replace_with_token(
            tokens.UNKNOWN
        ),
    )
    model.initialize()

    for block in self.blocks_with_throughput:
      self.assertTrue(model.validate_basic_block(block.block))
      self.assertTrue(model.validate_basic_blockTokens(block.block))
      self.assertTrue(model.validate_basic_block_with_throughput(block))

    block = basic_block.BasicBlock(
        basic_block.InstructionList(
            (basic_block.Instruction(mnemonic='FOOBAR'),)
        )
    )
    self.assertTrue(model.validate_basic_block(block))
    self.assertTrue(model.validate_basic_blockTokens(block))
    block_with_throughput = throughput.BasicBlockWithThroughput(
        block=block, throughputs=()
    )
    self.assertTrue(
        model.validate_basic_block_with_throughput(block_with_throughput)
    )


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
