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
"""Helper functions and classes for building models."""

import functools
from typing import Optional

import sonnet as snt
import tensorflow.compat.v1 as tf


class ResidualConnectionLayer(tf.keras.layers.Layer):
  """A Keras layer that implements residual connections.

  The layer takes a pair of tensors, assuming that the first tensor is the
  output of the subnetwork wrapped with a residual connection, and the second
  tensor is its input.

  Typical use:
    inputs = ...
    ops_outputs = some_tf_ops(inputs)
    residual_connection_layer = ResidualConnectionLayer(name='residual')
    outputs = residual_connection_layer((ops_outputs, inputs))

  TODO(ondrsaej): A cleaner design would work as a wrapper around another Keras
  layer. However, this would break compatibility with older trained models. As
  of 2021-01-18, we decided to use this design and keep the compatibility, but
  we should revisit it in the near future.
  """

  # @Override
  def build(self, layer_input_shapes: tuple[tf.TensorShape,
                                            tf.TensorShape]) -> None:
    output_shape, residual_shape = layer_input_shapes
    if output_shape.rank != 2:
      # NOTE(ondrasej): For simplicity, we require that the output has shape
      # (None, n) for some integer n. This requirement holds in virtually all
      # our use cases, since both Keras and Sonnet assume that all feature
      # vectors are one-dimensional.
      raise ValueError(
          'The rank of the output tensor must be 2, including the batch'
          f' dimension. Actual shape: {output_shape!r}')
    if not output_shape.is_compatible_with(residual_shape):
      # When the shapes of the input and the output of the subnetwork differ, we
      # add a learned linear transformation layer to match them.
      self._linear_transformation = tf.keras.layers.Dense(
          units=output_shape[1],
          activation=tf.keras.activations.linear,
          use_bias=False,
          name=self.name + '_transformation',
      )
      self._linear_transformation.build(residual_shape)
    else:
      # When the shapes are the same, we do not use the linear transformation.
      # According to https://arxiv.org/abs/1512.03385, it has almost no effect
      # on the training of the model when the shapes are the same.
      self._linear_transformation = None

  # @Override
  def call(self, layer_inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    output_part, residual_part = layer_inputs
    if self._linear_transformation:
      residual_part = self._linear_transformation(residual_part)

    return tf.math.add(output_part, residual_part, name=self.name)


def add_residual_connection(output_part: tf.Tensor,
                            residual_part: tf.Tensor,
                            name: Optional[str] = None) -> tf.Tensor:
  """Adds a residual connection to the output of a subnetwork.

  When the shape of `output_part` and `residual_part` are the same, then they
  are simply added, elementwise. If the shapes are different, the function
  creates a learned linear transformation layer that transforms the residual
  part to the right shape; in this case, the rank of the output part must be 2
  and the first dimension must be the batch dimension.

  Args:
    output_part: The tensor that contains the output of the subnetwork.
    residual_part: The input of the subnetwork.
    name: The name of the residual connection. This name is used for the
      tf.math.add operation that merges the two parts; if the linear
      transformation is used, its name is f'{name}_transformation'.

  Returns:
    A tensor that contains the output of the network merged with the residual
    connection.

  Raises:
    ValueError: If the rank of the output part is not two, including the batch
      dimension.
  """
  residual_layer = ResidualConnectionLayer(name=name)
  return residual_layer((output_part, residual_part))


def cast(dtype: tf.dtypes.DType) -> snt.AbstractModule:
  """Creates a sonnet module that casts a tensor to the specified dtype."""
  return snt.Module(build=functools.partial(tf.cast, dtype=dtype))
