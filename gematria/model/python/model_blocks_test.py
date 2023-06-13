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

import numpy as np
import tensorflow.compat.v1 as tf

from gematria.model.python import model_blocks


class ResidualConnectionLayerTest(tf.test.TestCase):

  def test_same_shapes(self):
    shape = (2, 4)
    dtype = tf.dtypes.float32
    input_tensor = tf.placeholder(shape=shape, dtype=dtype)
    residual_tensor = tf.placeholder(shape=shape, dtype=dtype)
    residual_layer = model_blocks.ResidualConnectionLayer(name='residual')
    output_tensor = residual_layer((input_tensor, residual_tensor))
    self.assertEqual(output_tensor.shape, shape)

    self.assertEmpty(residual_layer.weights)

    with self.session() as sess:
      input_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]],
                             dtype=dtype.as_numpy_dtype)
      residual_array = np.array([[-1, 1, -1, 1], [-2, 2, -2, 2]],
                                dtype=dtype.as_numpy_dtype)
      output_array = sess.run(
          output_tensor,
          feed_dict={
              input_tensor: input_array,
              residual_tensor: residual_array,
          },
      )
      self.assertAllEqual(output_array, [[0, 3, 2, 5], [3, 8, 5, 10]])

  def test_different_shapes(self):
    input_shape = (None, 2)
    residual_shape = (None, 3)
    dtype = tf.dtypes.float32
    input_tensor = tf.placeholder(shape=input_shape, dtype=dtype)
    residual_tensor = tf.placeholder(shape=residual_shape, dtype=dtype)
    residual_layer = model_blocks.ResidualConnectionLayer(name='residual')
    output_tensor = residual_layer((input_tensor, residual_tensor))
    output_tensor.shape.assert_is_compatible_with(input_shape)

    residual_layer_weights = residual_layer.weights
    self.assertLen(residual_layer_weights, 1)
    self.assertTrue(residual_layer_weights[0].shape.is_compatible_with((3, 2)))

    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      # In the first test, we use all zeros for the residual array. Any linear
      # projection must map this to zeros, thus the output must be the same as
      # input_array.
      input_array = np.array([[1, 2], [3, 4]], dtype=dtype.as_numpy_dtype)
      residual_array = np.zeros((2, 3), dtype=dtype.as_numpy_dtype)
      output_array = sess.run(
          output_tensor,
          feed_dict={
              input_tensor: input_array,
              residual_tensor: residual_array,
          },
      )
      self.assertAllEqual(output_array, input_array)

      # In the second test, we use a non-zero array. We can't test for exact
      # values because of the random initialization of the linear transformation
      # but with probability one, the output is different from the input.
      residual_array = np.ones((2, 3), dtype=dtype.as_numpy_dtype)
      output_array = sess.run(
          output_tensor,
          feed_dict={
              input_tensor: input_array,
              residual_tensor: residual_array,
          },
      )
      self.assertNotAllClose(output_array, input_array)


class AddResidualConnectioNTest(tf.test.TestCase):

  def test_same_shapes(self):
    shape = (2, 4)
    dtype = tf.dtypes.float32
    input_tensor = tf.placeholder(shape=shape, dtype=dtype)
    residual_tensor = tf.placeholder(shape=shape, dtype=dtype)
    output_tensor = model_blocks.add_residual_connection(
        input_tensor, residual_tensor, name='residual')
    self.assertEqual(output_tensor.shape, shape)

    with self.session() as sess:
      input_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]],
                             dtype=dtype.as_numpy_dtype)
      residual_array = np.array([[-1, 1, -1, 1], [-2, 2, -2, 2]],
                                dtype=dtype.as_numpy_dtype)
      output_array = sess.run(
          output_tensor,
          feed_dict={
              input_tensor: input_array,
              residual_tensor: residual_array,
          },
      )
      self.assertAllEqual(output_array, [[0, 3, 2, 5], [3, 8, 5, 10]])

  def test_different_shapes(self):
    input_shape = (None, 2)
    residual_shape = (None, 3)
    dtype = tf.dtypes.float32
    input_tensor = tf.placeholder(shape=input_shape, dtype=dtype)
    residual_tensor = tf.placeholder(shape=residual_shape, dtype=dtype)
    output_tensor = model_blocks.add_residual_connection(
        input_tensor, residual_tensor, name='residual')
    output_tensor.shape.assert_is_compatible_with(input_shape)

    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      # In the first test, we use all zeros for the residual array. Any linear
      # projection must map this to zeros, thus the output must be the same as
      # input_array.
      input_array = np.array([[1, 2], [3, 4]], dtype=dtype.as_numpy_dtype)
      residual_array = np.zeros((2, 3), dtype=dtype.as_numpy_dtype)
      output_array = sess.run(
          output_tensor,
          feed_dict={
              input_tensor: input_array,
              residual_tensor: residual_array,
          },
      )
      self.assertAllEqual(output_array, input_array)

      # In the second test, we use a non-zero array. We can't test for exact
      # values because of the random initialization of the linear transformation
      # but with probability one, the output is different from the input.
      residual_array = np.ones((2, 3), dtype=dtype.as_numpy_dtype)
      output_array = sess.run(
          output_tensor,
          feed_dict={
              input_tensor: input_array,
              residual_tensor: residual_array,
          },
      )
      self.assertNotAllClose(output_array, input_array)


class CastTest(tf.test.TestCase):

  def test_int32_to_float_cast(self):
    input_shape = (4, 24)
    input_dtype = tf.dtypes.int32
    output_dtype = tf.dtypes.float32
    input_tensor = tf.placeholder(shape=input_shape, dtype=input_dtype)
    cast = model_blocks.cast(output_dtype)
    output_tensor = cast(input_tensor)

    self.assertEqual(output_tensor.shape, input_shape)
    self.assertEqual(output_tensor.dtype, output_dtype)

    with self.session() as sess:
      input_array = np.ones(input_shape, input_dtype.as_numpy_dtype)
      output_array = sess.run(
          output_tensor, feed_dict={input_tensor: input_array})
      self.assertEqual(output_array.shape, input_shape)
      self.assertEqual(output_array.dtype, output_dtype.as_numpy_dtype)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
