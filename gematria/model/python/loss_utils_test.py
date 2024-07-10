# Copyright 2022 Google Inc.
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

from gematria.model.python import loss_utils
from gematria.model.python import options
import numpy as np
import tensorflow.compat.v1 as tf


class LossComputationTest(tf.test.TestCase):
  """The test for the LossComputation class."""

  def setUp(self):
    super().setUp()
    self.dtype = tf.float32
    self.numpy_dtype = self.dtype.as_numpy_dtype()
    self.actual_outputs_array = np.array(
        ((1,), (2,), (3,), (5,), (2.5,)), dtype=self.numpy_dtype
    )
    self.actual_outputs = tf.constant(self.actual_outputs_array)
    self.expected_outputs_array = np.array(
        ((4,), (2,), (3,), (1,), (2,)), dtype=self.numpy_dtype
    )
    self.expected_outputs = tf.constant(self.expected_outputs_array)
    self.full_mask = tf.ones_like(self.expected_outputs, dtype=tf.dtypes.bool)

    self.multitask_actual_outputs_array = np.array(
        ((1, 20), (2, 12), (3, 13), (5, 15), (2.5, 12.5)),
        dtype=self.numpy_dtype,
    )
    self.multitask_actual_outputs = tf.constant(
        self.multitask_actual_outputs_array
    )
    self.multitask_expected_outputs_array = np.array(
        ((4, 14.0), (2, 12.0), (3, 13.0), (1, 11.0), (2, 12.0)),
        dtype=self.numpy_dtype,
    )
    self.multitask_expected_outputs = tf.constant(
        self.multitask_expected_outputs_array
    )
    self.multitask_full_mask = tf.ones_like(
        self.multitask_expected_outputs, dtype=tf.dtypes.bool
    )

  def test_unscaled_loss(self):
    loss = loss_utils.LossComputation(
        self.actual_outputs,
        self.expected_outputs,
        self.full_mask,
        percentile_ranks=(0, 10, 50, 75, 100),
        dtype=self.dtype,
    )

    mse_tensor = loss.mean_squared_error
    mae_tensor = loss.mean_absolute_error
    huber_tensor = loss.loss_tensor(
        options.ErrorNormalization.NONE, options.LossType.HUBER
    )
    percentiles_tensor = loss.absolute_error_percentiles
    with self.session() as sess:
      mse, mae, huber, percentiles = sess.run(
          (mse_tensor, mae_tensor, huber_tensor, percentiles_tensor)
      )
      self.assertNear(
          float(mse), (3**2 + 0 + 0 + 4**2 + 0.5**2) / 5, 1e-6
      )
      self.assertNear(float(mae), (3 + 0 + 0 + 4 + 0.5) / 5, 1e-6)
      self.assertNear(
          float(huber), (2.5 + 0 + 0 + 3.5 + (0.5**2) / 2) / 5, 1e-6
      )
      self.assertAllEqual(percentiles, ((0,), (0,), (0.5,), (3,), (4,)))

  def test_percentage_loss(self):
    loss = loss_utils.LossComputation(
        self.actual_outputs,
        self.expected_outputs,
        self.full_mask,
        percentile_ranks=[25, 50, 60, 90],
        dtype=self.dtype,
    )

    mspe_tensor = loss.mean_squared_percentage_error
    mape_tensor = loss.mean_absolute_percentage_error
    percentiles_tensor = loss.absolute_percentage_error_percentiles
    with self.session() as sess:
      mspe, mape, percentiles = sess.run(
          (mspe_tensor, mape_tensor, percentiles_tensor)
      )
      self.assertAlmostEqual(
          float(mspe), ((3 / 4) ** 2 + 0 + 0 + 4**2 + (0.5 / 2) ** 2) / 5
      )
      self.assertAlmostEqual(float(mape), (3 / 4 + 0 + 0 + 4 + 0.5 / 2) / 5)
      self.assertAllEqual(percentiles, ((0,), (0.5 / 2,), (0.5 / 2,), (4,)))

  def test_normalized_loss_when_expected_value_greater_than_one(self):
    actual_outputs = tf.constant(
        ((1.3,), (-2,), (3,), (1,), (2,)), dtype=self.dtype
    )
    expected_outputs = tf.constant(
        ((1,), (4,), (3,), (0.5,), (0,)), dtype=self.dtype
    )
    mask = tf.ones_like(actual_outputs, dtype=tf.dtypes.bool)
    loss = loss_utils.LossComputation(
        actual_outputs, expected_outputs, mask, dtype=self.dtype
    )

    mean_absolute_error_tensor = loss.loss_tensor(
        options.ErrorNormalization.EXPECTED_VALUE_GREATER_THAN_ONE,
        options.LossType.MEAN_ABSOLUTE_ERROR,
    )
    mean_squared_error_tensor = loss.loss_tensor(
        options.ErrorNormalization.EXPECTED_VALUE_GREATER_THAN_ONE,
        options.LossType.MEAN_SQUARED_ERROR,
    )
    with self.session() as sess:
      mean_absolute_error = sess.run(mean_absolute_error_tensor)
      mean_squared_error = sess.run(mean_squared_error_tensor)
      self.assertAlmostEqual(
          float(mean_absolute_error), (0.3 + 1.5 + 0.0 + 0.5 + 2.0) / 5
      )
      self.assertAlmostEqual(
          float(mean_squared_error),
          (0.3**2 + 1.5**2 + 0.0 + 0.5**2 + 2.0**2) / 5,
          delta=1e-6,
      )

  def test_with_no_percentiles(self):
    loss = loss_utils.LossComputation(
        self.actual_outputs,
        self.expected_outputs,
        self.full_mask,
        dtype=self.dtype,
    )
    abs_percentiles_tensor = loss.absolute_error_percentiles
    pct_percentiles_tenros = loss.absolute_percentage_error_percentiles
    with self.session() as sess:
      abs_percentiles, pct_percentiles = sess.run(
          (abs_percentiles_tensor, pct_percentiles_tenros)
      )
      self.assertAllEqual(abs_percentiles, [])
      self.assertAllEqual(pct_percentiles, [])

  def test_multi_task(self):
    self.assertEqual(self.multitask_actual_outputs.shape, (5, 2))
    self.assertEqual(self.multitask_expected_outputs.shape, (5, 2))

    loss = loss_utils.LossComputation(
        self.multitask_actual_outputs,
        self.multitask_expected_outputs,
        self.multitask_full_mask,
        percentile_ranks=(0, 10, 50, 75, 100),
        dtype=self.dtype,
    )

    mse_tensor = loss.mean_squared_error
    mae_tensor = loss.mean_absolute_error
    huber_tensor = loss.loss_tensor(
        options.ErrorNormalization.NONE, options.LossType.HUBER
    )
    percentiles_tensor = loss.absolute_error_percentiles
    with self.session() as sess:
      mse, mae, huber, percentiles = sess.run(
          (mse_tensor, mae_tensor, huber_tensor, percentiles_tensor)
      )
      self.assertAllClose(
          mse,
          (
              (3**2 + 0 + 0 + 4**2 + 0.5**2) / 5,
              (6**2 + 0 + 0 + 4**2 + 0.5**2) / 5,
          ),
          1e-6,
      )
      self.assertAllClose(
          mae, ((3 + 0 + 0 + 4 + 0.5) / 5, (6 + 0 + 0 + 4 + 0.5) / 5), 1e-6
      )
      self.assertAllClose(
          huber,
          (
              (2.5 + 0 + 0 + 3.5 + (0.5**2) / 2) / 5,
              (5.5 + 0 + 0 + 3.5 + (0.5**2) / 2) / 5,
          ),
          1e-6,
      )
      self.assertAllEqual(
          percentiles, ((0, 0), (0, 0), (0.5, 0.5), (3, 4), (4, 6))
      )

  def test_multi_task_with_mask(self):
    loss = loss_utils.LossComputation(
        output_values=tf.constant(
            ((1, 20), (2, 12.1), (3, 100), (50, 150), (2.5, 12.5)),
            dtype=self.dtype,
        ),
        expected_outputs=tf.constant(
            ((4, 14), (500, 12), (30, 13), (1, 11), (2, 12)), dtype=self.dtype
        ),
        mask=tf.constant((
            (True, True),
            (False, True),
            (True, False),
            (False, False),
            (True, False),
        )),
        percentile_ranks=(0, 10, 50, 75, 100),
        dtype=self.dtype,
    )

    mse_tensor = loss.mean_squared_error
    mspe_tensor = loss.mean_squared_percentage_error
    mae_tensor = loss.mean_absolute_error
    mape_tensor = loss.mean_absolute_percentage_error
    huber_tensor = loss.loss_tensor(
        options.ErrorNormalization.NONE, options.LossType.HUBER
    )
    huber_percentage_tensor = loss.loss_tensor(
        options.ErrorNormalization.PERCENTAGE_ERROR, options.LossType.HUBER
    )
    percentile_tensor = loss.absolute_error_percentiles
    with self.session() as sess:
      mse, mspe, mae, mape, huber, huber_percentage, percentiles = sess.run((
          mse_tensor,
          mspe_tensor,
          mae_tensor,
          mape_tensor,
          huber_tensor,
          huber_percentage_tensor,
          percentile_tensor,
      ))
      self.assertAllClose(
          mse,
          ((3**2 + 27**2 + 0.5**2) / 3, (6**2 + 0.1**2) / 2),
          1e-6,
      )
      self.assertAllClose(
          mspe,
          (
              ((3 / 4) ** 2 + (27 / 30) ** 2 + (0.5 / 2) ** 2) / 3,
              ((6 / 14) ** 2 + (0.1 / 12) ** 2) / 2,
          ),
          1e-6,
      )
      self.assertAllClose(mae, ((3 + 27 + 0.5) / 3, (6 + 0.1) / 2), 1e-6)
      self.assertAllClose(
          mape, ((3 / 4 + 27 / 30 + 0.5 / 2) / 3, (6 / 14 + 0.1 / 12) / 2), 1e-6
      )
      self.assertAllClose(
          huber,
          ((2.5 + 26.5 + (0.5**2) / 2) / 3, (5.5 + (0.1**2) / 2) / 2),
          1e-6,
      )
      self.assertAllClose(
          huber_percentage,
          (
              (((3 / 4) ** 2) / 2 + ((27 / 30) ** 2) / 2 + ((0.5 / 2) ** 2) / 2)
              / 3,
              (((6 / 14) ** 2) / 2 + ((0.1 / 12) ** 2) / 2) / 2,
          ),
          1e-6,
      )
      self.assertAllClose(
          percentiles,
          ((0.5, 0.1), (0.5, 0.1), (3, 0.1), (27, 6), (27, 6)),
          1e-6,
      )

  def test_unknown_shape(self):
    actual_output = tf.placeholder(self.dtype, shape=(None, 1))
    expected_output = tf.placeholder(self.dtype, shape=(None, 1))
    mask = tf.placeholder(tf.dtypes.bool, shape=(None, 1))
    percentile_ranks = (0, 50, 75, 100)

    loss = loss_utils.LossComputation(
        actual_output,
        expected_output,
        mask,
        percentile_ranks=percentile_ranks,
        dtype=self.dtype,
    )

    mse_tensor = loss.mean_squared_error
    mae_tensor = loss.mean_absolute_error
    huber_tensor = loss.loss_tensor(
        options.ErrorNormalization.NONE, options.LossType.HUBER
    )
    percentiles_tensor = loss.absolute_error_percentiles

    self.assertEqual(mse_tensor.shape, (1,))
    self.assertEqual(mae_tensor.shape, (1,))
    self.assertEqual(huber_tensor.shape, (1,))
    self.assertEqual(percentiles_tensor.shape, (len(percentile_ranks), 1))

    feed_dict = {
        actual_output: self.actual_outputs_array,
        expected_output: self.expected_outputs_array,
        mask: np.ones_like(self.actual_outputs_array, dtype=bool),
    }
    with self.session() as sess:
      mse, mae, huber, percentiles = sess.run(
          (mse_tensor, mae_tensor, huber_tensor, percentiles_tensor),
          feed_dict=feed_dict,
      )
      self.assertNear(
          float(mse), (3**2 + 0 + 0 + 4**2 + 0.5**2) / 5, 1e-6
      )
      self.assertNear(float(mae), (3 + 0 + 0 + 4 + 0.5) / 5, 1e-6)
      self.assertNear(
          float(huber), (2.5 + 0 + 0 + 3.5 + (0.5**2) / 2) / 5, 1e-6
      )
      self.assertAllEqual(percentiles, ((0,), (0.5,), (3,), (4,)))

  def test_multi_task_unknown_shape(self):
    num_tasks = 2
    actual_output = tf.placeholder(self.dtype, shape=(None, num_tasks))
    expected_output = tf.placeholder(self.dtype, shape=(None, num_tasks))
    mask = tf.placeholder(tf.dtypes.bool, shape=(None, num_tasks))
    percentile_ranks = (0, 50, 75, 100)

    loss = loss_utils.LossComputation(
        actual_output,
        expected_output,
        mask,
        percentile_ranks=percentile_ranks,
        dtype=self.dtype,
    )

    mse_tensor = loss.mean_squared_error
    mae_tensor = loss.mean_absolute_error
    huber_tensor = loss.loss_tensor(
        options.ErrorNormalization.NONE, options.LossType.HUBER
    )
    percentiles_tensor = loss.absolute_error_percentiles

    self.assertEqual(mse_tensor.shape, (num_tasks,))
    self.assertEqual(mae_tensor.shape, (num_tasks,))
    self.assertEqual(huber_tensor.shape, (num_tasks,))
    self.assertEqual(
        percentiles_tensor.shape, (len(percentile_ranks), num_tasks)
    )

    feed_dict = {
        actual_output: self.multitask_actual_outputs_array,
        expected_output: self.multitask_expected_outputs_array,
        mask: np.ones_like(self.multitask_actual_outputs_array, dtype=bool),
    }
    with self.session() as sess:
      mse, mae, huber, percentiles = sess.run(
          (mse_tensor, mae_tensor, huber_tensor, percentiles_tensor),
          feed_dict=feed_dict,
      )
      self.assertNDArrayNear(
          mse,
          np.array((
              (3**2 + 0 + 0 + 4**2 + 0.5**2) / 5,
              (6**2 + 0 + 0 + 4**2 + 0.5**2) / 5,
          )),
          1e-6,
      )
      self.assertNDArrayNear(
          mae, (((3 + 0 + 0 + 4 + 0.5) / 5, (6 + 0 + 0 + 4 + 0.5) / 5)), 1e-6
      )
      self.assertNDArrayNear(
          huber,
          ((
              (2.5 + 0 + 0 + 3.5 + (0.5**2) / 2) / 5,
              (5.5 + 0 + 0 + 3.5 + (0.5**2) / 2) / 5,
          )),
          1e-6,
      )
      self.assertAllEqual(percentiles, ((0, 0), (0.5, 0.5), (3, 4), (4, 6)))

  def test_single_task_unknown_shape(self):
    num_tasks = 1
    actual_output = tf.placeholder(self.dtype, shape=(None, num_tasks))
    expected_output = tf.placeholder(self.dtype, shape=(None, num_tasks))
    mask = tf.ones_like(actual_output, tf.dtypes.bool)
    percentile_ranks = (0, 50, 75, 100)

    loss = loss_utils.LossComputation(
        actual_output,
        expected_output,
        mask,
        percentile_ranks=percentile_ranks,
        dtype=self.dtype,
    )

    mse_tensor = loss.mean_squared_error
    mae_tensor = loss.mean_absolute_error
    huber_tensor = loss.loss_tensor(
        options.ErrorNormalization.NONE, options.LossType.HUBER
    )
    percentiles_tensor = loss.absolute_error_percentiles

    self.assertEqual(mse_tensor.shape, (num_tasks,))
    self.assertEqual(mae_tensor.shape, (num_tasks,))
    self.assertEqual(huber_tensor.shape, (num_tasks,))
    self.assertEqual(
        percentiles_tensor.shape, (len(percentile_ranks), num_tasks)
    )

    feed_dict = {
        actual_output: self.actual_outputs_array.reshape((-1, 1)),
        expected_output: self.expected_outputs_array.reshape((-1, 1)),
    }
    with self.session() as sess:
      mse, mae, huber, percentiles = sess.run(
          (mse_tensor, mae_tensor, huber_tensor, percentiles_tensor),
          feed_dict=feed_dict,
      )
      self.assertNear(
          float(mse), (3**2 + 0 + 0 + 4**2 + 0.5**2) / 5, 1e-6
      )
      self.assertNear(float(mae), (3 + 0 + 0 + 4 + 0.5) / 5, 1e-6)
      self.assertNear(
          float(huber), (2.5 + 0 + 0 + 3.5 + (0.5**2) / 2) / 5, 1e-6
      )
      self.assertAllEqual(percentiles, ((0,), (0.5,), (3,), (4,)))


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
