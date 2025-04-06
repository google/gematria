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
import tf_keras


class ResidualConnectionLayer(tf_keras.layers.Layer):
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
  def __init__(
      self, layer_input_shapes: tuple[tf.TensorShape, tf.TensorShape], **kwargs
  ) -> None:
    super().__init__(**kwargs)
    output_shape, residual_shape = layer_input_shapes
    if output_shape.rank != 2:
      # NOTE(ondrasej): For simplicity, we require that the output has shape
      # (None, n) for some integer n. This requirement holds in virtually all
      # our use cases, since both Keras and Sonnet assume that all feature
      # vectors are one-dimensional.
      raise ValueError(
          'The rank of the output tensor must be 2, including the batch'
          f' dimension. Actual shape: {output_shape!r}'
      )
    if not output_shape.is_compatible_with(residual_shape):
      # When the shapes of the input and the output of the subnetwork differ, we
      # add a learned linear transformation layer to match them.
      self._linear_transformation = tf_keras.layers.Dense(
          units=output_shape[1],
          activation=tf_keras.activations.linear,
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
