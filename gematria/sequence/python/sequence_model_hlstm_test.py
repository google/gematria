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
from gematria.model.python import oov_token_behavior
from gematria.sequence.python import sequence_model_hlstm
from gematria.testing.python import model_test
import tensorflow as tf
import tf_keras

_OutOfVocabularyTokenBehavior = oov_token_behavior.OutOfVocabularyTokenBehavior


class SequenceModelHlstmTest(parameterized.TestCase, model_test.TestCase):

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2num(self, loss_type, loss_normalization):
    model = sequence_model_hlstm.HierarchicalLstmModel(
        learning_rate=0.01,
        use_deltas=False,
        loss_type=loss_type,
        loss_normalization=loss_normalization,
        token_embedding_size=32,
        instruction_embedding_size=36,
        block_embedding_size=40,
        bidirectional=False,
        output_layers=(),
        task_output_layers=(),
        tokens=self.tokens,
        dtype=tf.dtypes.float32,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
    )
    model.initialize()
    self.check_training_model(model, self.blocks_with_throughput)

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2seq(self, loss_type, loss_normalization):
    model = sequence_model_hlstm.HierarchicalLstmModel(
        learning_rate=0.01,
        use_deltas=True,
        loss_type=loss_type,
        loss_normalization=loss_normalization,
        token_embedding_size=32,
        instruction_embedding_size=36,
        block_embedding_size=40,
        bidirectional=False,
        output_layers=(),
        task_output_layers=(),
        tokens=self.tokens,
        dtype=tf.dtypes.float32,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
    )
    model.initialize()
    self.check_training_model(model, self.blocks_with_throughput)

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2num_multi_task(self, loss_type, loss_normalization):
    model = sequence_model_hlstm.HierarchicalLstmModel(
        learning_rate=0.01,
        use_deltas=False,
        task_list=('llvm', 'test'),
        loss_type=loss_type,
        loss_normalization=loss_normalization,
        token_embedding_size=32,
        instruction_embedding_size=36,
        block_embedding_size=40,
        bidirectional=False,
        output_layers=(16, 832),
        task_output_layers=(17,),
        tokens=self.tokens,
        dtype=tf.dtypes.float32,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
    )
    model.initialize()
    self.assertIsInstance(model._model._block_lstm, tf_keras.layers.LSTM)
    self.check_training_model(model, self.blocks_with_throughput, num_epochs=40)

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2num_bidirectional(self, loss_type, loss_normalization):
    model = sequence_model_hlstm.HierarchicalLstmModel(
        learning_rate=0.01,
        use_deltas=False,
        loss_type=loss_type,
        loss_normalization=loss_normalization,
        token_embedding_size=32,
        instruction_embedding_size=36,
        block_embedding_size=40,
        bidirectional=True,
        output_layers=(16, 832),
        task_output_layers=(17,),
        tokens=self.tokens,
        dtype=tf.dtypes.float32,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
    )
    model.initialize()
    self.assertIsInstance(
        model._model._block_lstm, tf_keras.layers.Bidirectional
    )
    self.check_training_model(model, self.blocks_with_throughput, num_epochs=40)

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2seq_bidirectional(self, loss_type, loss_normalization):
    model = sequence_model_hlstm.HierarchicalLstmModel(
        learning_rate=0.01,
        use_deltas=True,
        loss_type=loss_type,
        loss_normalization=loss_normalization,
        token_embedding_size=32,
        instruction_embedding_size=36,
        block_embedding_size=40,
        bidirectional=True,
        output_layers=(16, 832),
        task_output_layers=(17,),
        tokens=self.tokens,
        dtype=tf.dtypes.float32,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
    )
    model.initialize()
    self.check_training_model(model, self.blocks_with_throughput, num_epochs=40)


if __name__ == '__main__':
  tf.test.main()
