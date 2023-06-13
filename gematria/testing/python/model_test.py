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
"""Provides a base class for testing Gematria models."""

import math

from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from gematria.model.python import options
from gematria.testing.python import basic_blocks_with_throughput

FLAGS = flags.FLAGS

# Shortcuts for types from options.
_ErrorNormalization = options.ErrorNormalization
_LossType = options.LossType

# List of possible combinations of loss types and error normalization in loss
# computation. This tuple can be used as an argument list for
# absl.testing.parameterized.named_parameters().
#
# Example:
#   @parameterized.named_parameters(*LOSS_TYPES_AND_LOSS_NORMALIZATIONS)
#   def test_train_a_model(self, loss_type, loss_normalization):
#     self.do_the_actual_testing(loss_type, loss_normalization)
LOSS_TYPES_AND_LOSS_NORMALIZATIONS = (
    ('mse - none', _LossType.MEAN_SQUARED_ERROR, _ErrorNormalization.NONE),
    ('mae - none', _LossType.MEAN_ABSOLUTE_ERROR, _ErrorNormalization.NONE),
    ('huber - none', _LossType.HUBER, _ErrorNormalization.NONE),
    (
        'mse - percentage',
        _LossType.MEAN_SQUARED_ERROR,
        _ErrorNormalization.PERCENTAGE_ERROR,
    ),
    (
        'mae - percentage',
        _LossType.MEAN_ABSOLUTE_ERROR,
        _ErrorNormalization.PERCENTAGE_ERROR,
    ),
    (
        'huber - percentage',
        _LossType.HUBER,
        _ErrorNormalization.PERCENTAGE_ERROR,
    ),
    (
        'mse - greater_than_one',
        _LossType.MEAN_SQUARED_ERROR,
        _ErrorNormalization.EXPECTED_VALUE_GREATER_THAN_ONE,
    ),
    (
        'mae - greater_than_one',
        _LossType.MEAN_ABSOLUTE_ERROR,
        _ErrorNormalization.EXPECTED_VALUE_GREATER_THAN_ONE,
    ),
    (
        'huber - greater_than_one',
        _LossType.HUBER,
        _ErrorNormalization.EXPECTED_VALUE_GREATER_THAN_ONE,
    ),
)

OPTIMIZER_TYPES = (
    ('adam', options.OptimizerType.ADAM),
    ('sgd', options.OptimizerType.SGD),
    ('rmsprop', options.OptimizerType.RMSPROP),
)

RNN_TYPES_AND_BIDIRECTIONAL_STATE = (
    ('LSTM_BIDIRECTIONAL', options.RnnType.LSTM, True),
    ('RNN_BIDIRECTIONAL', options.RnnType.GRU, True),
    ('LSTM_UNIDIRECTIONAL', options.RnnType.LSTM, False),
    ('RNN_UNIDIRECTIONAL', options.RnnType.GRU, False),
)


class TestCase(basic_blocks_with_throughput.TestCase, tf.test.TestCase):
  """Base class for testing Gematria models.

  Gets basic blocks from the test data library and provides a method for testing
  training of a model.
  """

  # By default, the tests use only a single basic blocks.
  num_blocks = 1

  def setUp(self):
    """Initializes the test.

    Performs common initialization tasks needed by the tests:
      1. Sets the TensorFlow random number generator seed according to the test
         command-line flag to ensure deterministically random tests.
      2. Loads basic blocks from test data. The number of blocks can be adjusted
         by overriding setUp() in the child class and setting self.num_blocks
         before calling the setUp() method of the superclass.
    """
    super().setUp()

    tf.random.set_random_seed(FLAGS.test_random_seed)

  def check_training_model(
      self,
      model,
      blocks=None,
      num_epochs=20,
      max_expected_min_loss=0.2,
      log_directory=None,
      print_output_to_log=False,
      session=None,
  ):
    """Tests training the given model.

    Runs model.train_batch() on 'blocks' for the given number of epochs and
    tracks the minimal loss observed during the training. Verifies that the
    minimal loss is less or equal to 'expected_min_loss'.

    Args:
      model: The tested model. The method assumes that the model is initialized.
      blocks: The basic blocks used for training the model. If None, data from
        self.blocks_with_throughput are used.
      num_epochs: The number of training epochs to run.
      max_expected_min_loss: The maximal allowed value for the minimal loss seen
        during the training.
      log_directory: Optional directory that is used to store event logs
        (summaries) that can be analyzed using TensorBoard. When None, the logs
        are not stored.
      print_output_to_log: When True, the contents of the output tensor is
        printed to the log at each step.
      session: An optional session to run the training in. If `session` is not
        None, the method will run the training in it, but it will not release
        the session at the end. If `session` is None, the function will create a
        session just for the training, and it will release it at the end.
    """
    blocks = blocks or self.blocks_with_throughput

    def _check_training(sess):
      if log_directory is not None:
        tf.summary.FileWriter(logdir=log_directory, graph=sess.graph)
      sess.run(tf.global_variables_initializer())
      schedule = model.schedule_batch(blocks)

      # The loss at the end of the training may increase temporarily, and it is
      # thus not stable. Instead, we use the minimal loss from all epochs.
      min_loss = math.inf
      min_mse = [math.inf] * model.num_tasks
      min_relative_mse = [math.inf] * model.num_tasks
      for epoch in range(num_epochs):
        stats = model.train_batch(sess, schedule)
        if print_output_to_log:
          output = sess.run(model.output_tensor, schedule)
          # The output is a 2D tensor of shape (batch_size, num_tasks). When
          # num_tasks == 1, the output is a 2D column tensor. We reshape it to
          # (num_tasks,), so that it prints on a single line. When num_tasks > 1
          # we leave the shape as is.
          if output.shape[1] == 1:
            output = output.reshape((-1,))
          logging.info('Output: %r', output)
        # Check basic properties.
        self.assertEqual(stats.epoch, epoch + 1)
        self.assertGreaterEqual(stats.loss, 0)
        self.assertAllGreaterEqual(stats.absolute_mse, 0)
        self.assertAllGreaterEqual(stats.relative_mse, 0)

        # Check the percentiles.
        num_percentiles = len(model.collected_percentile_ranks)
        self.assertEqual(stats.percentile_ranks,
                         model.collected_percentile_ranks)
        self.assertLen(stats.absolute_error_percentiles, num_percentiles)
        self.assertLen(stats.relative_error_percentiles, num_percentiles)

        min_loss = min(min_loss, stats.loss)
        min_mse = np.minimum(min_mse, stats.absolute_mse)
        min_relative_mse = np.minimum(min_relative_mse, stats.relative_mse)
        logging.info(
            'epoch = %d, min_loss = %f, min_mse = %r, min_relative_mse = %r',
            epoch,
            min_loss,
            min_mse.tolist(),
            min_relative_mse.tolist(),
        )
      self.assertAllLess(min_relative_mse, max_expected_min_loss)

    if session:
      # If an external session was provided, just run the training in the
      # session and assume that the owner will take care of releasing it
      # afterwards.
      _check_training(session)
    else:
      # Otherwise, create a session for this call and release it at the end.
      with self.session() as session:
        _check_training(session)
