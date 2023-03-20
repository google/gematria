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
import tensorflow.compat.v1 as tf

from gematria.basic_block.python import tokens
from gematria.granite.python import rnn_token_model
from gematria.model.python import oov_token_behavior
from gematria.model.python import options
from gematria.testing.python import model_test

_OutOfVocabularyTokenBehavior = oov_token_behavior.OutOfVocabularyTokenBehavior


class RnnTokenModelTest(parameterized.TestCase, model_test.TestCase):

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2num(self, loss_type, loss_normalization):
    num_message_passing_iterations = 1
    node_embedding_size = 8
    edge_embedding_size = 6
    global_embedding_size = 4
    node_update_layers = (16,)
    edge_update_layers = (18,)
    global_update_layers = (13,)
    readout_layers = (16,)
    rnn_output_size = 32
    task_readout_layers = ()
    model = rnn_token_model.RnnTokenModel(
        tokens=self.tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
        use_deltas=False,
        learning_rate=0.01,
        loss_type=loss_type,
        loss_normalization=loss_normalization,
        node_embedding_size=node_embedding_size,
        edge_embedding_size=edge_embedding_size,
        global_embedding_size=global_embedding_size,
        node_update_layers=node_update_layers,
        edge_update_layers=edge_update_layers,
        global_update_layers=global_update_layers,
        readout_layers=readout_layers,
        task_readout_layers=task_readout_layers,
        rnn_output_size=rnn_output_size,
        rnn_dropout=0.1,
        rnn_type=options.RnnType.LSTM,
        rnn_bidirectional=False,
        graph_module_layer_normalization=False,
        task_readout_input_layer_normalization=False,
        readout_input_layer_normalization=False,
        num_message_passing_iterations=num_message_passing_iterations,
        dtype=tf.dtypes.float32,
    )
    model.initialize()
    self.check_training_model(model, self.blocks_with_throughput)

  @parameterized.named_parameters(*model_test.RNN_TYPES_AND_BIDIRECTIONAL_STATE)
  def test_train_rnn_type(self, rnn_type, bidirectional):
    num_message_passing_iterations = 1
    node_embedding_size = 8
    edge_embedding_size = 6
    global_embedding_size = 4
    node_update_layers = (16,)
    edge_update_layers = (18,)
    global_update_layers = (13,)
    readout_layers = (16,)
    rnn_output_size = 32
    task_readout_layers = ()
    model = rnn_token_model.RnnTokenModel(
        tokens=self.tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
        use_deltas=False,
        learning_rate=0.01,
        loss_type=options.LossType.MEAN_SQUARED_ERROR,
        loss_normalization=options.ErrorNormalization.NONE,
        node_embedding_size=node_embedding_size,
        edge_embedding_size=edge_embedding_size,
        global_embedding_size=global_embedding_size,
        node_update_layers=node_update_layers,
        edge_update_layers=edge_update_layers,
        global_update_layers=global_update_layers,
        readout_layers=readout_layers,
        task_readout_layers=task_readout_layers,
        rnn_output_size=rnn_output_size,
        rnn_dropout=0.1,
        rnn_type=rnn_type,
        rnn_bidirectional=bidirectional,
        graph_module_layer_normalization=False,
        task_readout_input_layer_normalization=False,
        readout_input_layer_normalization=False,
        num_message_passing_iterations=num_message_passing_iterations,
        dtype=tf.dtypes.float32,
    )
    model.initialize()
    self.check_training_model(model, self.blocks_with_throughput)

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2seq(self, loss_type, loss_normalization):
    num_message_passing_iterations = 1
    node_embedding_size = 8
    edge_embedding_size = 6
    global_embedding_size = 4
    node_update_layers = (16,)
    edge_update_layers = (18,)
    global_update_layers = (13,)
    readout_layers = (16,)
    rnn_output_size = 32
    task_readout_layers = ()
    model = rnn_token_model.RnnTokenModel(
        tokens=self.tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
        use_deltas=True,
        learning_rate=0.01,
        loss_type=loss_type,
        loss_normalization=loss_normalization,
        node_embedding_size=node_embedding_size,
        edge_embedding_size=edge_embedding_size,
        global_embedding_size=global_embedding_size,
        node_update_layers=node_update_layers,
        edge_update_layers=edge_update_layers,
        global_update_layers=global_update_layers,
        readout_layers=readout_layers,
        task_readout_layers=task_readout_layers,
        rnn_output_size=rnn_output_size,
        rnn_dropout=0.1,
        rnn_type=options.RnnType.LSTM,
        rnn_bidirectional=False,
        graph_module_layer_normalization=False,
        task_readout_input_layer_normalization=False,
        readout_input_layer_normalization=False,
        num_message_passing_iterations=num_message_passing_iterations,
        dtype=tf.dtypes.float32,
    )
    model.initialize()
    self.check_training_model(model, self.blocks_with_throughput)

  def test_train_seq2num_multi_task(self):
    num_message_passing_iterations = 1
    node_embedding_size = 8
    edge_embedding_size = 6
    global_embedding_size = 4
    node_update_layers = (16,)
    edge_update_layers = (18,)
    global_update_layers = (13,)
    readout_layers = (16,)
    rnn_output_size = 32
    task_readout_layers = ()
    model = rnn_token_model.RnnTokenModel(
        tokens=self.tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
        use_deltas=False,
        task_list=('task1', 'task2'),
        learning_rate=0.01,
        node_embedding_size=node_embedding_size,
        edge_embedding_size=edge_embedding_size,
        global_embedding_size=global_embedding_size,
        node_update_layers=node_update_layers,
        edge_update_layers=edge_update_layers,
        global_update_layers=global_update_layers,
        readout_layers=readout_layers,
        task_readout_layers=task_readout_layers,
        rnn_output_size=rnn_output_size,
        rnn_dropout=0.1,
        rnn_type=options.RnnType.LSTM,
        rnn_bidirectional=False,
        graph_module_layer_normalization=False,
        task_readout_input_layer_normalization=False,
        readout_input_layer_normalization=False,
        num_message_passing_iterations=num_message_passing_iterations,
        dtype=tf.dtypes.float32,
    )
    model.initialize()
    self.check_training_model(model, self.blocks_with_throughput)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
