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
from gematria.basic_block.python import tokens
from gematria.granite.python import token_graph_builder_model
from gematria.model.python import model_base
from gematria.model.python import model_blocks
from gematria.model.python import oov_token_behavior
from gematria.testing.python import model_test
import tensorflow.compat.v1 as tf
import tf_keras as keras

_OutOfVocabularyTokenBehavior = oov_token_behavior.OutOfVocabularyTokenBehavior

_RESIDUAL_CONNECTION_LAYER_CLASS = (
    'gematria.model.python.model_blocks.ResidualConnectionLayer'
)


class TokenGraphBuilderModelTest(parameterized.TestCase, model_test.TestCase):

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2num(self, loss_type, loss_normalization):
    num_message_passing_iterations = 1
    node_embedding_size = 14
    edge_embedding_size = 16
    global_embedding_size = 18
    node_update_layers = (15, 7)
    edge_update_layers = (13, 9)
    global_update_layers = (11, 17)
    readout_layers = (16,)
    task_readout_layers = ()
    activation = functools.partial(keras.activations.relu, alpha=0.1)
    model = token_graph_builder_model.TokenGraphBuilderModel(
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
        readout_activation=activation,
        update_activation=activation,
        graph_module_layer_normalization=True,
        task_readout_input_layer_normalization=False,
        readout_input_layer_normalization=False,
        num_message_passing_iterations=num_message_passing_iterations,
        dtype=tf.dtypes.float32,
    )
    with mock.patch(
        'tf_keras.layers.Dense',
        side_effect=keras.layers.Dense,
    ) as mock_dense:
      model.initialize()
    mock_dense.assert_has_calls(
        (
            mock.call(
                16, activation=mock.ANY, bias_initializer='glorot_normal'
            ),
            mock.call(1, activation=keras.activations.linear, use_bias=False),
        )
    )

    self.check_training_model(model, self.blocks_with_throughput, num_epochs=40)

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2seq(self, loss_type, loss_normalization):
    num_message_passing_iterations = 1
    node_embedding_size = 14
    edge_embedding_size = 16
    global_embedding_size = 18
    node_update_layers = (15, 7)
    edge_update_layers = (13, 9)
    global_update_layers = (11, 17)
    readout_layers = (16,)
    task_readout_layers = ()
    model = token_graph_builder_model.TokenGraphBuilderModel(
        tokens=self.tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
        use_deltas=True,
        use_delta_loss=False,
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
        graph_module_layer_normalization=True,
        task_readout_input_layer_normalization=False,
        readout_input_layer_normalization=False,
        num_message_passing_iterations=num_message_passing_iterations,
        dtype=tf.dtypes.float32,
    )
    model.initialize()
    self.check_training_model(model, self.blocks_with_throughput, num_epochs=40)

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_traing_seq2seq_multi_task(self, loss_type, loss_normalization):
    num_message_passing_iterations = 1
    node_embedding_size = 14
    edge_embedding_size = 16
    global_embedding_size = 18
    node_update_layers = (15, 7)
    edge_update_layers = (13, 9)
    global_update_layers = (11, 17)
    readout_layers = (16,)
    task_readout_layers = (17, 18)
    model = token_graph_builder_model.TokenGraphBuilderModel(
        tokens=self.tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
        use_deltas=True,
        use_delta_loss=False,
        task_list=('task1', 'task2'),
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
        graph_module_layer_normalization=False,
        task_readout_input_layer_normalization=False,
        readout_input_layer_normalization=False,
        num_message_passing_iterations=num_message_passing_iterations,
        dtype=tf.dtypes.float32,
    )
    # NOTE(ondrasej): The single  block data set used in this test is too small
    # to see the effects of layer normalization. Instead, we intercept the
    # constructors of layer normalization layers in Keras and Sonnet; the Sonet
    # version is used only with graph_module_layer_normalization, and the Keras
    # version is used with task_readout_input_layer_normalization and
    # readout_input_layer_normalization. In this test, all of them are False,
    # and so none of these functions should be used.
    with mock.patch(
        'tf_keras.layers.LayerNormalization',
        side_effect=keras.layers.LayerNormalization,
    ) as keras_layer_norm:
      model.initialize()
    keras_layer_norm.assert_not_called()
    self.check_training_model(model, self.blocks_with_throughput, num_epochs=40)

  def test_train_seq2seq_readout_layer_norm(self):
    num_message_passing_iterations = 1
    node_embedding_size = 14
    edge_embedding_size = 16
    global_embedding_size = 18
    node_update_layers = (15, 7)
    edge_update_layers = (13, 9)
    global_update_layers = (11, 17)
    readout_layers = (16,)
    task_readout_layers = ()
    model = token_graph_builder_model.TokenGraphBuilderModel(
        tokens=self.tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
        use_deltas=True,
        use_delta_loss=False,
        learning_rate=0.01,
        node_embedding_size=node_embedding_size,
        edge_embedding_size=edge_embedding_size,
        global_embedding_size=global_embedding_size,
        node_update_layers=node_update_layers,
        edge_update_layers=edge_update_layers,
        global_update_layers=global_update_layers,
        readout_layers=readout_layers,
        task_readout_layers=task_readout_layers,
        graph_module_layer_normalization=False,
        task_readout_input_layer_normalization=False,
        readout_input_layer_normalization=True,
        num_message_passing_iterations=num_message_passing_iterations,
        dtype=tf.dtypes.float32,
    )
    # In this test, only readout_input_layer_normalization is True, so we expect
    # to see the Keras version used once.
    with mock.patch(
        'tf_keras.layers.LayerNormalization',
        side_effect=keras.layers.LayerNormalization,
    ) as keras_layer_norm:
      model.initialize()
    keras_layer_norm.assert_called_once_with(
        name='readout_input_layer_normalization'
    )
    self.check_training_model(model, self.blocks_with_throughput, num_epochs=40)

  @parameterized.named_parameters(
      *model_test.LOSS_TYPES_AND_LOSS_NORMALIZATIONS
  )
  def test_train_seq2seq_task_readout_layer_norm(
      self, loss_type, loss_normalization
  ):
    num_message_passing_iterations = 1
    node_embedding_size = 14
    edge_embedding_size = 16
    global_embedding_size = 18
    node_update_layers = (15, 7)
    edge_update_layers = (13, 9)
    global_update_layers = (11, 17)
    readout_layers = (16,)
    task_readout_layers = ()
    model = token_graph_builder_model.TokenGraphBuilderModel(
        tokens=self.tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
        use_deltas=True,
        use_delta_loss=False,
        learning_rate=0.01,
        node_embedding_size=node_embedding_size,
        edge_embedding_size=edge_embedding_size,
        global_embedding_size=global_embedding_size,
        node_update_layers=node_update_layers,
        edge_update_layers=edge_update_layers,
        global_update_layers=global_update_layers,
        readout_layers=readout_layers,
        task_readout_layers=task_readout_layers,
        graph_module_layer_normalization=False,
        task_readout_input_layer_normalization=True,
        readout_input_layer_normalization=False,
        num_message_passing_iterations=num_message_passing_iterations,
        dtype=tf.dtypes.float32,
    )
    # In this test, only task_readout_input_layer_normalization is True, so we
    # expect to see the Keras version used once.
    with mock.patch(
        'tf_keras.layers.LayerNormalization',
        side_effect=keras.layers.LayerNormalization,
    ) as keras_layer_norm:
      model.initialize()
    keras_layer_norm.assert_called_once_with(
        name='task_readout_input_layer_normalization'
    )
    self.check_training_model(model, self.blocks_with_throughput, num_epochs=40)

  def test_train_seq2seq_graph_module_layer_norm(self):
    num_message_passing_iterations = 1
    node_embedding_size = 14
    edge_embedding_size = 16
    global_embedding_size = 18
    node_update_layers = (15, 7)
    edge_update_layers = (13, 9)
    global_update_layers = (11, 17)
    readout_layers = (16,)
    task_readout_layers = (15,)
    model = token_graph_builder_model.TokenGraphBuilderModel(
        tokens=self.tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
        use_deltas=True,
        use_delta_loss=False,
        learning_rate=0.01,
        node_embedding_size=node_embedding_size,
        edge_embedding_size=edge_embedding_size,
        global_embedding_size=global_embedding_size,
        node_update_layers=node_update_layers,
        edge_update_layers=edge_update_layers,
        global_update_layers=global_update_layers,
        readout_layers=readout_layers,
        task_readout_layers=task_readout_layers,
        graph_module_layer_normalization=True,
        task_readout_input_layer_normalization=False,
        readout_input_layer_normalization=False,
        num_message_passing_iterations=num_message_passing_iterations,
        dtype=tf.dtypes.float32,
    )
    # In this test, only graph_module_layer_normalization is True, thus only the
    # Sonnet version of layer normalization should be used.
    with mock.patch(
        'tf_keras.layers.LayerNormalization',
        side_effect=keras.layers.LayerNormalization,
    ) as keras_layer_norm:
      model.initialize()
    keras_layer_norm.assert_has_calls(
        (
            mock.call(name='graph_network_layer_norm_1_0_nodes'),
            mock.call(name='graph_network_layer_norm_1_0_edges'),
            mock.call(name='graph_network_layer_norm_1_0_globals'),
        )
    )
    self.check_training_model(model, self.blocks_with_throughput, num_epochs=40)

  def test_train_seq2seq_with_sent_edges(self):
    num_message_passing_iterations = 1
    node_embedding_size = 14
    edge_embedding_size = 16
    global_embedding_size = 18
    node_update_layers = (15, 7)
    edge_update_layers = (13, 9)
    global_update_layers = (11, 17)
    readout_layers = (16,)
    task_readout_layers = (15,)
    model = token_graph_builder_model.TokenGraphBuilderModel(
        tokens=self.tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
        use_deltas=True,
        use_delta_loss=False,
        learning_rate=0.01,
        node_embedding_size=node_embedding_size,
        edge_embedding_size=edge_embedding_size,
        global_embedding_size=global_embedding_size,
        node_update_layers=node_update_layers,
        edge_update_layers=edge_update_layers,
        global_update_layers=global_update_layers,
        readout_layers=readout_layers,
        task_readout_layers=task_readout_layers,
        graph_module_layer_normalization=True,
        task_readout_input_layer_normalization=False,
        readout_input_layer_normalization=False,
        use_sent_edges=True,
        num_message_passing_iterations=num_message_passing_iterations,
        dtype=tf.dtypes.float32,
    )
    # In this test, only graph_module_layer_normalization is True, thus only the
    # Sonnet version of layer normalization should be used.
    with mock.patch(
        'tf_keras.layers.LayerNormalization',
        side_effect=keras.layers.LayerNormalization,
    ) as keras_layer_norm:
      model.initialize()
    keras_layer_norm.assert_has_calls(
        (
            mock.call(name='graph_network_layer_norm_1_0_nodes'),
            mock.call(name='graph_network_layer_norm_1_0_edges'),
            mock.call(name='graph_network_layer_norm_1_0_globals'),
        )
    )
    self.check_training_model(model, self.blocks_with_throughput, num_epochs=40)

  def test_train_seq2seq_with_readout_residual_connection(self):
    num_message_passing_iterations = 1
    node_embedding_size = 14
    edge_embedding_size = 16
    global_embedding_size = 18
    node_update_layers = (15, 7)
    edge_update_layers = (13, 9)
    global_update_layers = (11, 17)
    readout_layers = (16,)
    task_readout_layers = (15,)
    model = token_graph_builder_model.TokenGraphBuilderModel(
        tokens=self.tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
        use_deltas=True,
        use_delta_loss=False,
        learning_rate=0.01,
        node_embedding_size=node_embedding_size,
        edge_embedding_size=edge_embedding_size,
        global_embedding_size=global_embedding_size,
        node_update_layers=node_update_layers,
        edge_update_layers=edge_update_layers,
        global_update_layers=global_update_layers,
        readout_layers=readout_layers,
        task_readout_layers=task_readout_layers,
        graph_module_layer_normalization=True,
        task_readout_input_layer_normalization=False,
        readout_input_layer_normalization=False,
        readout_residual_connections=True,
        use_sent_edges=True,
        num_message_passing_iterations=num_message_passing_iterations,
        dtype=tf.dtypes.float32,
    )
    # In this test, only graph_module_layer_normalization is True, thus only the
    # Sonnet version of layer normalization should be used.
    with mock.patch(
        _RESIDUAL_CONNECTION_LAYER_CLASS,
        side_effect=model_blocks.ResidualConnectionLayer,
    ) as residual_connection_layer:
      model.initialize()
    self.assertEqual(
        residual_connection_layer.call_args_list,
        [mock.call(name='readout_residual_connections')],
    )
    self.check_training_model(model, self.blocks_with_throughput, num_epochs=40)

  def test_train_seq2seq_with_task_readout_residual_connection(self):
    num_message_passing_iterations = 1
    node_embedding_size = 14
    edge_embedding_size = 16
    global_embedding_size = 18
    node_update_layers = (15, 7)
    edge_update_layers = (13, 9)
    global_update_layers = (11, 17)
    readout_layers = (16,)
    task_readout_layers = (15,)
    model = token_graph_builder_model.TokenGraphBuilderModel(
        tokens=self.tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
        use_deltas=True,
        use_delta_loss=False,
        learning_rate=0.01,
        node_embedding_size=node_embedding_size,
        edge_embedding_size=edge_embedding_size,
        global_embedding_size=global_embedding_size,
        node_update_layers=node_update_layers,
        edge_update_layers=edge_update_layers,
        global_update_layers=global_update_layers,
        readout_layers=readout_layers,
        task_readout_layers=task_readout_layers,
        graph_module_layer_normalization=False,
        task_readout_input_layer_normalization=True,
        readout_input_layer_normalization=False,
        task_readout_residual_connections=True,
        use_sent_edges=True,
        num_message_passing_iterations=num_message_passing_iterations,
        dtype=tf.dtypes.float32,
    )
    # In this test, only graph_module_layer_normalization is True, thus only the
    # Sonnet version of layer normalization should be used.
    with mock.patch(
        _RESIDUAL_CONNECTION_LAYER_CLASS,
        side_effect=model_blocks.ResidualConnectionLayer,
    ) as residual_connection_layer:
      model.initialize()
    self.assertEqual(
        residual_connection_layer.call_args_list,
        [mock.call(name='task_readout_residual_connections')],
    )
    self.check_training_model(model, self.blocks_with_throughput, num_epochs=40)

  def test_train_seq2seq_with_replacement_token(self):
    # A list of tokens that contains all the "helper" tokens used by the graph
    # builder but no tokens for the actual assembly code. Transforming a
    # non-empty basic block with a graph builder that uses only these tokens is
    # guaranteed to trigger the out-of-vocabulary behavior.
    structural_tokens = (
        tokens.ADDRESS,
        tokens.IMMEDIATE,
        tokens.MEMORY,
        tokens.UNKNOWN,
    )
    num_message_passing_iterations = 1
    node_embedding_size = 14
    edge_embedding_size = 16
    global_embedding_size = 18
    node_update_layers = (15, 7)
    edge_update_layers = (13, 9)
    global_update_layers = (11, 17)
    readout_layers = (16,)
    task_readout_layers = (15,)
    model = token_graph_builder_model.TokenGraphBuilderModel(
        tokens=structural_tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        out_of_vocabulary_behavior=(
            _OutOfVocabularyTokenBehavior.replace_with_token(tokens.UNKNOWN)
        ),
        use_deltas=True,
        use_delta_loss=False,
        learning_rate=0.01,
        node_embedding_size=node_embedding_size,
        edge_embedding_size=edge_embedding_size,
        global_embedding_size=global_embedding_size,
        node_update_layers=node_update_layers,
        edge_update_layers=edge_update_layers,
        global_update_layers=global_update_layers,
        readout_layers=readout_layers,
        task_readout_layers=task_readout_layers,
        graph_module_layer_normalization=False,
        task_readout_input_layer_normalization=True,
        readout_input_layer_normalization=False,
        task_readout_residual_connections=True,
        use_sent_edges=True,
        num_message_passing_iterations=num_message_passing_iterations,
        dtype=tf.dtypes.float32,
    )
    model.initialize()
    self.check_training_model(model, self.blocks_with_throughput, num_epochs=40)

  def test_train_seq2seq_returning_error_on_unknown_token(self):
    # A list of tokens that contains all the "helper" tokens used by the graph
    # builder but no tokens for the actual assembly code. Transforming a
    # non-empty basic block with a graph builder that uses only these tokens is
    # guaranteed to trigger the out-of-vocabulary behavior.
    structural_tokens = (
        tokens.ADDRESS,
        tokens.IMMEDIATE,
        tokens.MEMORY,
        tokens.UNKNOWN,
    )
    num_message_passing_iterations = 1
    node_embedding_size = 14
    edge_embedding_size = 16
    global_embedding_size = 18
    node_update_layers = (15, 7)
    edge_update_layers = (13, 9)
    global_update_layers = (11, 17)
    readout_layers = (16,)
    task_readout_layers = (15,)
    model = token_graph_builder_model.TokenGraphBuilderModel(
        tokens=structural_tokens,
        immediate_token=tokens.IMMEDIATE,
        fp_immediate_token=tokens.IMMEDIATE,
        address_token=tokens.ADDRESS,
        memory_token=tokens.MEMORY,
        out_of_vocabulary_behavior=_OutOfVocabularyTokenBehavior.return_error(),
        use_deltas=True,
        use_delta_loss=False,
        learning_rate=0.01,
        node_embedding_size=node_embedding_size,
        edge_embedding_size=edge_embedding_size,
        global_embedding_size=global_embedding_size,
        node_update_layers=node_update_layers,
        edge_update_layers=edge_update_layers,
        global_update_layers=global_update_layers,
        readout_layers=readout_layers,
        task_readout_layers=task_readout_layers,
        graph_module_layer_normalization=False,
        task_readout_input_layer_normalization=True,
        readout_input_layer_normalization=False,
        task_readout_residual_connections=True,
        use_sent_edges=True,
        num_message_passing_iterations=num_message_passing_iterations,
        dtype=tf.dtypes.float32,
    )
    model.initialize()
    with self.assertRaises(model_base.AddBasicBlockError):
      model.schedule_batch(self.blocks_with_throughput)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
