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

from gematria.model.python import model_blocks
import tensorflow as tf


class ResidualConnectionLayerTest(tf.test.TestCase):

  def test_same_shapes(self):
    shape = tf.TensorShape((2, 4))
    dtype = tf.dtypes.float32
    residual_layer = model_blocks.ResidualConnectionLayer(
        layer_input_shapes=(shape, shape), name='residual'
    )

    self.assertEmpty(residual_layer.weights)

    input_array = tf.constant(
        [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype.as_numpy_dtype
    )
    residual_array = tf.constant(
        [[-1, 1, -1, 1], [-2, 2, -2, 2]], dtype=dtype.as_numpy_dtype
    )
    output = residual_layer((input_array, residual_array))
    self.assertEqual(output.shape, shape)
    self.assertAllEqual(output, [[0, 3, 2, 5], [3, 8, 5, 10]])

  def test_different_shapes(self):
    input_shape = tf.TensorShape((None, 2))
    residual_shape = tf.TensorShape((None, 3))
    dtype = tf.dtypes.float32
    residual_layer = model_blocks.ResidualConnectionLayer(
        layer_input_shapes=(input_shape, residual_shape), name='residual'
    )

    residual_layer_weights = residual_layer.weights
    self.assertLen(residual_layer_weights, 1)
    self.assertTrue(residual_layer_weights[0].shape.is_compatible_with((3, 2)))

    # In the first test, we use all zeros for the residual array. Any linear
    # projection must map this to zeros, thus the output must be the same as
    # input_tensor.
    input_tensor = tf.constant([[1, 2], [3, 4]], dtype=dtype)
    residual_tensor = tf.zeros((2, 3), dtype=dtype)
    output_tensor = residual_layer((input_tensor, residual_tensor))
    self.assertAllEqual(output_tensor, input_tensor)

    # In the second test, we use a non-zero array. We can't test for exact
    # values because of the random initialization of the linear transformation
    # but with probability one, the output is different from the input.
    residual_tensor = tf.ones((2, 3), dtype=dtype.as_numpy_dtype)
    output_tensor = residual_layer((input_tensor, residual_tensor))
    self.assertNotAllClose(output_tensor, input_tensor)


if __name__ == '__main__':
  tf.test.main()
