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
"""Contains types, classes and functions for loss computation in Gematria."""

# TensorFlow Probability depends on `distutils`, which is no longer included
# with Python 3.12 and onwards. `distutils` is instead provided by `setuptools`,
# but the `sys.path` modification enabling this is not correctly supported by
# the `rules_python` directory structure.
# Including `setuptools` here should allow us to continue using `tfp`.
# TODO(vbshah): Remove this import once
# https://github.com/bazel-contrib/rules_python/issues/2071 is resolved.
import setuptools  # pylint: disable=unused-import

from collections.abc import Sequence

from gematria.model.python import options
import tensorflow as tf
import tensorflow_probability as tfp


def _huber(
    normalized_absolute_errors: tf.Tensor,
) -> tf.Tensor:
  """Computes the Huber loss from normalized absolute errors."""
  # The delta parameter from the Huber loss definition.
  huber_delta = tf.constant(1.0, dtype=normalized_absolute_errors.dtype)
  # The expression in the quadratic part of the Huber loss expression.
  # It is squared in the return statement below.
  quadratic = tf.minimum(normalized_absolute_errors, huber_delta)
  # The linear part of the Huber loss expression. This is zero when
  # absolute_error <= huber_delta.
  linear = normalized_absolute_errors - quadratic
  return 0.5 * tf.square(quadratic) + huber_delta * linear


def _make_percentile_tensor(
    values: tf.Tensor,
    num_tasks: int,
    percentile_ranks: Sequence[int],
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
  """Creates a percentile tensor from 'values' using self.percentile_ranks.

  Args:
    values: A 2D ragged tensor from which the percentiles are collected. The
      percentiles are collected along the axis 0 of `values`.
    num_tasks: The number of tasks for the current model. Used for validation
      purposes.
    percentile_ranks: The percentile ranks to use for calculating the tensor.

  Returns:
    Percentiles based on self_percentile_ranks and the values. The returned
    tensor is of shape (N_PERCENTILE_RANKS, T), where T is the number of
    tasks.
  """
  if not percentile_ranks:
    return tf.constant([], dtype=dtype)

  percentile_tensors = []
  # NOTE(ondrasej): As of Nov 2022, tfp.stats.percentile() is not compatible
  # with ragged tensors, so we need to split the ragged tensor into rows and
  # then stack the individual percentile tensors to the desired output shape.
  for task in range(num_tasks):
    task_values = values[task]
    percentile_tensors.append(
        tfp.stats.percentile(task_values, percentile_ranks)
    )
    assert percentile_tensors[-1].shape.is_compatible_with((None,))
  percentile_tensor = tf.stack(percentile_tensors, axis=1)
  assert not percentile_ranks or percentile_tensor.shape.is_compatible_with(
      (len(percentile_ranks), num_tasks)
  )
  return percentile_tensor


@tf.autograph.experimental.do_not_convert
def _apply_loss_function(
    loss_type: options.LossType,
    normalized_delta: tf.Tensor,
) -> tf.Tensor:
  """Applies the selected loss function to normalized absolute deltas."""
  match loss_type:
    case options.LossType.MEAN_SQUARED_ERROR:
      return tf.square(normalized_delta)
    case options.LossType.MEAN_ABSOLUTE_ERROR:
      return tf.abs(normalized_delta)
    case options.LossType.HUBER:
      return _huber(tf.abs(normalized_delta))
  raise ValueError(f'Unexpected loss type {loss_type!r}')


@tf.autograph.experimental.do_not_convert
def _apply_normalization(
    normalization: options.ErrorNormalization,
    delta: tf.Tensor,
    expected_outputs: tf.Tensor,
) -> tf.Tensor:
  """Applies the given error normalization function to deltas.

  Args:
    normalization: The selected normalization function.
    delta: The deltas between the actual output and the expected output.
    expected_outputs: The expected output values.

  Returns:
    A tensor that contains the normalized error value for each input.
  """
  match normalization:
    case options.ErrorNormalization.NONE:
      return delta
    case options.ErrorNormalization.PERCENTAGE_ERROR:
      return delta / expected_outputs
    case options.ErrorNormalization.EXPECTED_VALUE_GREATER_THAN_ONE:
      return delta / tf.math.maximum(
          expected_outputs,
          tf.ones_like(expected_outputs, dtype=expected_outputs.dtype),
      )
  raise ValueError(f'Unknown normalization {normalization!r}')


def _loss_tensor(
    normalization: options.ErrorNormalization,
    loss_type: options.LossType,
    num_tasks: int,
    delta: tf.Tensor,
    expected_outputs: tf.Tensor,
) -> tf.Tensor:
  """Creates a loss tensor for the given loss type.

  Args:
    normalization: The error normalization used in the loss tensor.
    loss_type: The loss function used in the loss tensor.
    num_tasks: The number of tasks in the input data.
    delta: A tensor that contains the absolute differences between actual and
      expected outputs. Must have have shape (None, num_tasks).
    expected_outputs: A tensor that contains the expected outputs. Must have
      shape (None, num_tasks).

  Returns:
    A tensor that contains the loss computed with the selected loss function
    and normalization. Has shape (num_tasks, ).
  """
  normalized_delta = _apply_normalization(
      normalization=normalization,
      delta=delta,
      expected_outputs=expected_outputs,
  )
  normalized_delta.shape.assert_is_compatible_with((num_tasks, None))
  loss_values = _apply_loss_function(
      loss_type=loss_type,
      normalized_delta=normalized_delta,
  )
  loss = tf.reduce_mean(loss_values, axis=1)
  loss.shape.assert_is_compatible_with((num_tasks,))
  return loss


class LossComputation(tf.experimental.ExtensionType):
  """The result of loss computation."""

  loss_tensor: tf.Tensor
  mean_absolute_error: tf.Tensor
  mean_squared_error: tf.Tensor
  mean_absolute_percentage_error: tf.Tensor
  mean_squared_percentage_error: tf.Tensor
  absolute_error_percentiles: tf.Tensor
  absolute_percentage_error_percentiles: tf.Tensor


def create(
    output_values: tf.Tensor,
    expected_outputs: tf.Tensor,
    mask: tf.Tensor,
    dtype: tf.dtypes.DType,
    percentile_ranks: Sequence[int] = (),
    normalization: options.ErrorNormalization = options.ErrorNormalization.NONE,
    loss_type: options.LossType = options.LossType.MEAN_SQUARED_ERROR,
) -> LossComputation:
  """Initializes the loss computation.

  Args:
    output_values: The actual outputs of the model; of shape (N, T) where N is
      the number of samples and T is the number of tasks.
    expected_outputs: The expected outputs of the model; of shape (N, T) where
      N is the number of samples and T is the number of tasks.
    mask: The mask for well defined outputs; of shape (N, T) where N is the
      number of samples and T is the number of tasks. The loss includes only
      outputs where the corresponding entry of the mask is True.
    dtype: The TensorFlow DType used by the model.
    percentile_ranks: The percentile ranks used in the error statistics. These
      must be integers between 0 and 100.
    normalization: The normalization to apply when calculating the loss.
    loss_type: THe type of loss to compute.

  Returns: A LossComputation object containing the calculated losses.
  """
  if len(output_values.shape) != 2:
    raise ValueError(
        'output_values must be a 2D tensor. Actual shape:'
        f' {output_values.shape}'
    )
  if not expected_outputs.shape.is_compatible_with(output_values.shape):
    raise ValueError(
        'Expected expected_outputs.shape to be compatible with '
        f'{output_values.shape}. Found {expected_outputs.shape}'
    )

  num_tasks = output_values.shape[1] or expected_outputs.shape[1]
  assert num_tasks is not None
  if not mask.shape.is_compatible_with(output_values.shape):
    raise ValueError(
        'Expected mask.shape to be compatible with'
        f' {output_values.shape}. Found {mask.shape}'
    )
  if tf.dtypes.bool != mask.dtype:
    raise ValueError(
        f'Expected mask.dtype to be tf.dtypes.bool. Found {mask.dtype}.'
    )

  # tf.ragged.boolean_mask() does not have an `axis` argument to control which
  # dimension is ragged and in case of 2D tensors it is always the second one.
  # We transpose the data so that the first (non-ragged) dimension goes along
  # tasks, and the second (ragged) dimension goes along the values.
  # All the tensors below have the shape
  mask = tf.transpose(mask)
  output_values = tf.ragged.boolean_mask(tf.transpose(output_values), mask)
  assert output_values.shape.is_compatible_with((num_tasks, None))
  expected_outputs = tf.ragged.boolean_mask(
      tf.transpose(expected_outputs), mask
  )
  assert expected_outputs.shape.is_compatible_with((num_tasks, None))

  absolute_errors = tf.abs(output_values - expected_outputs)
  assert absolute_errors.shape.is_compatible_with((num_tasks, None))

  absolute_percentage_errors = absolute_errors / expected_outputs
  assert absolute_percentage_errors.shape.is_compatible_with((num_tasks, None))

  return LossComputation(
      loss_tensor=_loss_tensor(
          normalization,
          loss_type,
          num_tasks,
          absolute_errors,
          expected_outputs,
      ),
      mean_absolute_error=_loss_tensor(
          options.ErrorNormalization.NONE,
          options.LossType.MEAN_ABSOLUTE_ERROR,
          num_tasks,
          absolute_errors,
          expected_outputs,
      ),
      mean_squared_error=_loss_tensor(
          options.ErrorNormalization.NONE,
          options.LossType.MEAN_SQUARED_ERROR,
          num_tasks,
          absolute_errors,
          expected_outputs,
      ),
      mean_absolute_percentage_error=_loss_tensor(
          options.ErrorNormalization.PERCENTAGE_ERROR,
          options.LossType.MEAN_ABSOLUTE_ERROR,
          num_tasks,
          absolute_errors,
          expected_outputs,
      ),
      mean_squared_percentage_error=_loss_tensor(
          options.ErrorNormalization.PERCENTAGE_ERROR,
          options.LossType.MEAN_SQUARED_ERROR,
          num_tasks,
          absolute_errors,
          expected_outputs,
      ),
      absolute_error_percentiles=_make_percentile_tensor(
          absolute_errors, num_tasks, percentile_ranks, dtype
      ),
      absolute_percentage_error_percentiles=_make_percentile_tensor(
          absolute_percentage_errors, num_tasks, percentile_ranks, dtype
      ),
  )
