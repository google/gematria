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

import os

from absl.testing import parameterized
from gematria.model.python import training
from gematria.testing.python import basic_blocks_with_throughput
import numpy as np
import tensorflow as tf


class TrainingEpochStatsTest(tf.test.TestCase):
  """The test case for TrainingEpochStats."""

  def test_initialize_seq2num(self):
    epoch = 123
    loss = 0.1234
    absolute_mse = np.array((234,))
    relative_mse = np.array((0.123,))
    relative_mae = np.array((12,))
    percentile_ranks = [50, 90]
    absolute_error_percentiles = np.array(((1.0,), (2.0,)))
    relative_error_percentiles = np.array(((0.3,), (0.7,)))
    stats = training.TrainingEpochStats(
        epoch=epoch,
        loss=loss,
        absolute_mse=absolute_mse,
        relative_mse=relative_mse,
        relative_mae=relative_mae,
        percentile_ranks=percentile_ranks,
        absolute_error_percentiles=np.array(absolute_error_percentiles),
        relative_error_percentiles=np.array(relative_error_percentiles),
    )
    self.assertEqual(stats.epoch, epoch)
    self.assertEqual(stats.loss, loss)
    self.assertAllEqual(stats.absolute_mse, absolute_mse)
    self.assertAllEqual(stats.relative_mse, relative_mse)
    self.assertAllEqual(stats.relative_mae, relative_mae)
    self.assertEqual(stats.percentile_ranks, percentile_ranks)
    self.assertAllEqual(
        stats.absolute_error_percentiles, absolute_error_percentiles
    )
    self.assertAllEqual(
        stats.relative_error_percentiles, relative_error_percentiles
    )
    self.assertIsNone(stats.absolute_delta_mse)
    self.assertIsNone(stats.absolute_delta_mae)
    self.assertIsNone(stats.absolute_delta_error_percentiles)

  def test_initialize_seq2num_multitask(self):
    epoch = 123
    loss = 0.1234
    absolute_mse = np.array((234, 345))
    relative_mse = np.array((0.123, 0.234))
    relative_mae = np.array((12, 46))
    percentile_ranks = [50, 90]
    absolute_error_percentiles = np.array(((1.0, 2.0), (2.0, 3.0)))
    relative_error_percentiles = np.array(((0.3, 0.7), (0.7, 1.5)))
    stats = training.TrainingEpochStats(
        epoch=epoch,
        loss=loss,
        absolute_mse=absolute_mse,
        relative_mse=relative_mse,
        relative_mae=relative_mae,
        percentile_ranks=percentile_ranks,
        absolute_error_percentiles=np.array(absolute_error_percentiles),
        relative_error_percentiles=np.array(relative_error_percentiles),
    )
    self.assertEqual(stats.epoch, epoch)
    self.assertEqual(stats.loss, loss)
    self.assertAllEqual(stats.absolute_mse, absolute_mse)
    self.assertAllEqual(stats.relative_mse, relative_mse)
    self.assertAllEqual(stats.relative_mae, relative_mae)
    self.assertEqual(stats.percentile_ranks, percentile_ranks)
    self.assertAllEqual(
        stats.absolute_error_percentiles, absolute_error_percentiles
    )
    self.assertAllEqual(
        stats.relative_error_percentiles, relative_error_percentiles
    )
    self.assertIsNone(stats.absolute_delta_mse)
    self.assertIsNone(stats.absolute_delta_mae)
    self.assertIsNone(stats.absolute_delta_error_percentiles)

  def test_initialize_seq2seq(self):
    epoch = 123
    loss = 0.222
    absolute_mse = np.array((234,))
    relative_mse = np.array((0.345,))
    relative_mae = np.array((0.2345,))
    percentile_ranks = [30, 60, 80]
    absolute_error_percentiles = np.array(((1.2,), (3.4,), (5.6,)))
    relative_error_percentiles = np.array(((0.01,), (0.23,), (0.34,)))
    absolute_delta_mse = np.array((456,))
    absolute_delta_mae = np.array((36,))
    absolute_delta_error_percentiles = np.array(((6.7,), (7.8,), (8.9,)))
    stats = training.TrainingEpochStats(
        epoch=epoch,
        loss=loss,
        percentile_ranks=percentile_ranks,
        absolute_mse=absolute_mse,
        relative_mse=relative_mse,
        relative_mae=relative_mae,
        absolute_error_percentiles=absolute_error_percentiles,
        relative_error_percentiles=relative_error_percentiles,
        absolute_delta_mse=absolute_delta_mse,
        absolute_delta_mae=absolute_delta_mae,
        absolute_delta_error_percentiles=absolute_delta_error_percentiles,
    )
    self.assertEqual(stats.epoch, epoch)
    self.assertEqual(stats.loss, loss)
    self.assertAllEqual(stats.absolute_mse, absolute_mse)
    self.assertAllEqual(stats.relative_mse, relative_mse)
    self.assertAllEqual(stats.relative_mae, relative_mae)
    self.assertAllEqual(stats.absolute_delta_mse, absolute_delta_mse)
    self.assertAllEqual(
        stats.absolute_error_percentiles, absolute_error_percentiles
    )
    self.assertAllEqual(
        stats.relative_error_percentiles, relative_error_percentiles
    )
    self.assertAllEqual(
        stats.absolute_delta_error_percentiles, absolute_delta_error_percentiles
    )

  def test_initialize_with_invalid_shapes(self):
    with self.assertRaisesRegex(ValueError, 'relative_mse'):
      training.TrainingEpochStats(
          epoch=1,
          loss=1.0,
          absolute_mse=np.array((1.0,)),
          relative_mse=np.array((2.0, 3.0)),
          relative_mae=np.array((0.5,)),
          percentile_ranks=[50, 90],
          absolute_error_percentiles=np.array(((1.0,), (2.0,))),
          relative_error_percentiles=np.array(((0.1,), (0.2,))),
      )
    with self.assertRaisesRegex(ValueError, 'relative_mae'):
      training.TrainingEpochStats(
          epoch=1,
          loss=1.0,
          absolute_mse=np.array((1.0, 2.0)),
          relative_mse=np.array((2.0, 3.0)),
          relative_mae=np.array((0.5,)),
          percentile_ranks=[50, 90],
          absolute_error_percentiles=np.array(((1.0, 2.0), (2.0, 3.0))),
          relative_error_percentiles=np.array(((0.1, 0.2), (0.2, 0.3))),
      )
    with self.assertRaisesRegex(ValueError, 'absolute_delta_mse'):
      training.TrainingEpochStats(
          epoch=1,
          loss=1.0,
          absolute_mse=np.array((1.0, 2.0)),
          relative_mse=np.array((2.0, 3.0)),
          relative_mae=np.array((0.5, 0.7)),
          absolute_delta_mse=np.array((4.0, 5.0, 5.5)),
          absolute_delta_mae=np.array((6.0, 7.0)),
          percentile_ranks=[50, 90],
          absolute_error_percentiles=np.array(((1.0, 2.0), (2.0, 3.0))),
          absolute_delta_error_percentiles=np.array(((4.0, 5.0), (6.0, 7.0))),
          relative_error_percentiles=np.array(((0.1, 0.2), (0.2, 0.3))),
      )
    with self.assertRaisesRegex(ValueError, 'absolute_delta_mae'):
      training.TrainingEpochStats(
          epoch=1,
          loss=1.0,
          absolute_mse=np.array((1.0, 2.0)),
          relative_mse=np.array((2.0, 3.0)),
          relative_mae=np.array((0.5, 0.7)),
          absolute_delta_mse=np.array((4.0, 5.0)),
          absolute_delta_mae=np.array((6.0, 7.0, 8.0)),
          percentile_ranks=[50, 90],
          absolute_error_percentiles=np.array(((1.0, 2.0), (2.0, 3.0))),
          absolute_delta_error_percentiles=np.array(((4.0, 5.0), (6.0, 7.0))),
          relative_error_percentiles=np.array(((0.1, 0.2), (0.2, 0.3))),
      )

  def test_initialize_with_invalid_percentiles(self):
    with self.assertRaisesRegex(ValueError, 'absolute_error_percentiles'):
      # Invalid stats - absolute error percentile count mismatch.
      training.TrainingEpochStats(
          epoch=1,
          loss=1.0,
          absolute_mse=np.array((1.0,)),
          relative_mse=np.array((2.0,)),
          relative_mae=np.array((0.5,)),
          percentile_ranks=[50, 90],
          absolute_error_percentiles=np.array(((1.0,), (2.0,), (3.0,))),
          relative_error_percentiles=np.array(((0.1,), (0.2,))),
      )
    with self.assertRaisesRegex(ValueError, 'relative_error_percentiles'):
      # Invalid stats - relative error percentile count mismatch.
      training.TrainingEpochStats(
          epoch=1,
          loss=1.0,
          absolute_mse=np.array((1.0,)),
          relative_mse=np.array((2.0,)),
          relative_mae=np.array((0.5,)),
          percentile_ranks=[50, 90],
          absolute_error_percentiles=np.array(((1.0,), (2.0,))),
          relative_error_percentiles=np.array(((0.1,), (0.2,), (0.3,))),
      )
    with self.assertRaisesRegex(ValueError, 'absolute_delta_error_percentiles'):
      # Invalid stats - absolute delta error percentile count mismatch.
      training.TrainingEpochStats(
          epoch=3,
          loss=1.0,
          absolute_mse=np.array((1.0,)),
          relative_mse=np.array((2.0,)),
          relative_mae=np.array((0.5,)),
          absolute_delta_mse=np.array((4.0,)),
          absolute_delta_mae=np.array((2.0,)),
          percentile_ranks=[5, 10],
          absolute_error_percentiles=np.array(((0.1,), (0.2,))),
          relative_error_percentiles=np.array(((0.3,), (0.4,))),
          absolute_delta_error_percentiles=np.array(((0.5,), (0.6,), (0.61,))),
      )

  def test_initialize_with_incomplete_seq2seq_stats(self):
    with self.assertRaisesRegex(ValueError, 'absolute_delta_mse'):
      # Invalid stats - absolute_delta_mse is missing.
      training.TrainingEpochStats(
          epoch=3,
          loss=1.0,
          absolute_mse=np.array((1.0,)),
          relative_mse=np.array((2.0,)),
          relative_mae=np.array((0.5,)),
          absolute_delta_mae=np.array((0.2,)),
          percentile_ranks=[5, 10],
          absolute_error_percentiles=np.array(((0.1,), (0.2,))),
          relative_error_percentiles=np.array(((0.3,), (0.4,))),
          absolute_delta_error_percentiles=np.array(((0.5,), (0.6,))),
      )
    with self.assertRaisesRegex(ValueError, 'absolute_delta_mae'):
      # Invalid stats - absolute_delta_mae is missing.
      training.TrainingEpochStats(
          epoch=3,
          loss=1.0,
          absolute_mse=1.0,
          relative_mse=np.array((2.0,)),
          relative_mae=np.array((0.5,)),
          absolute_delta_mse=np.array((0.1,)),
          percentile_ranks=[5, 10],
          absolute_error_percentiles=np.array(((0.1,), (0.2,))),
          relative_error_percentiles=np.array(((0.3,), (0.4,))),
          absolute_delta_error_percentiles=np.array(((0.5,), (0.6,))),
      )
    with self.assertRaisesRegex(ValueError, 'absolute_delta_error_percentiles'):
      # Invalid stats - absolute_delta_error_percentiles is missing.
      training.TrainingEpochStats(
          epoch=3,
          loss=1.0,
          absolute_mse=1.0,
          relative_mse=np.array((2.0,)),
          relative_mae=np.array((0.5,)),
          absolute_delta_mse=np.array((0.1,)),
          absolute_delta_mae=np.array((0.2,)),
          percentile_ranks=[5, 10],
          absolute_error_percentiles=np.array(((0.1,), (0.2,))),
          relative_error_percentiles=np.array(((0.3,), (0.4,))),
      )

  def test_string_conversion(self):
    stats = training.TrainingEpochStats(
        epoch=123,
        loss=345,
        absolute_mse=np.array((234,)),
        relative_mae=np.array((0.2,)),
        relative_mse=np.array((0.1,)),
        percentile_ranks=[50, 90],
        absolute_error_percentiles=np.array(((5.0,), (9.0,))),
        relative_error_percentiles=np.array(((0.5,), (0.9,))),
    )
    expected_output = (
        'epoch: 123, loss: 345\n'
        'absolute mse: [234], 50%: [5.0], 90%: [9.0]\n'
        'relative mse: [0.1], mae: [0.2], 50%: [0.5], 90%: [0.9]'
    )
    self.assertEqual(str(stats), expected_output)

  def test_string_conversion_multitask(self):
    stats = training.TrainingEpochStats(
        epoch=123,
        loss=345,
        absolute_mse=np.array((234, 235)),
        relative_mae=np.array((0.2, 0.3)),
        relative_mse=np.array((0.1, 0.2)),
        percentile_ranks=[50, 75, 90],
        absolute_error_percentiles=np.array(
            ((5.0, 6.0), (9.0, 10.0), (13.0, 14.0))
        ),
        relative_error_percentiles=np.array(
            ((0.5, 0.6), (0.9, 1.0), (1.3, 1.4))
        ),
    )
    expected_outputs = (
        'epoch: 123, loss: 345\n'
        'absolute mse: [234, 235], 50%: [5.0, 6.0], 75%: [9.0, 10.0],'
        ' 90%: [13.0, 14.0]\n'
        'relative mse: [0.1, 0.2], mae: [0.2, 0.3], 50%: [0.5, 0.6],'
        ' 75%: [0.9, 1.0], 90%: [1.3, 1.4]'
    )
    self.assertEqual(str(stats), expected_outputs)

  def test_string_conversion_no_percentiles(self):
    stats = training.TrainingEpochStats(
        epoch=123,
        loss=345,
        absolute_mse=np.array((234,)),
        relative_mae=np.array((0.2,)),
        relative_mse=np.array((0.1,)),
        percentile_ranks=[],
        absolute_error_percentiles=np.array(()),
        relative_error_percentiles=np.array(()),
    )
    expected_output = (
        'epoch: 123, loss: 345\n'
        'absolute mse: [234]\n'
        'relative mse: [0.1], mae: [0.2]'
    )
    self.assertEqual(str(stats), expected_output)

  def test_string_conversion_with_seq2seq_stats(self):
    stats = training.TrainingEpochStats(
        epoch=123,
        loss=345,
        absolute_mse=np.array((234,)),
        relative_mae=np.array((0.2,)),
        relative_mse=np.array((0.1,)),
        percentile_ranks=[50, 90],
        absolute_error_percentiles=np.array(((5.0,), (9.0,))),
        relative_error_percentiles=np.array(((0.5,), (0.9,))),
        absolute_delta_mse=np.array((321,)),
        absolute_delta_mae=np.array((23,)),
        absolute_delta_error_percentiles=np.array(((0.3,), (0.4,))),
    )
    expected_output = (
        'epoch: 123, loss: 345\n'
        'absolute mse: [234], 50%: [5.0], 90%: [9.0]\n'
        'relative mse: [0.1], mae: [0.2], 50%: [0.5], 90%: [0.9]\n'
        'absolute delta mse: [321], mae: [23], 50%: [0.3], 90%: [0.4]'
    )
    self.assertEqual(str(stats), expected_output)


class BatchesTest(
    parameterized.TestCase,
    basic_blocks_with_throughput.TestCase,
    tf.test.TestCase,
):
  """Tests for the Batches() function."""

  num_blocks = 10

  def _get_blocks(self, with_throughput):
    return self.blocks_with_throughput if with_throughput else self.blocks

  def _get_num_instructions_callback(self, with_throughput):
    return (
        training.get_num_instructions_in_block_with_throughput
        if with_throughput
        else training.get_num_instructions_in_block
    )

  @parameterized.named_parameters(
      ('with throughput', True), ('without throughput', False)
  )
  def test_no_limit(self, with_throughput):
    blocks = self._get_blocks(with_throughput)
    get_num_instructions = self._get_num_instructions_callback(with_throughput)
    batches = tuple(training.batches(blocks, get_num_instructions))
    self.assertSequenceEqual(batches, (blocks,))

  @parameterized.named_parameters(
      ('with throughput', True), ('without throughput', False)
  )
  def test_max_blocks_limit(self, with_throughput):
    blocks = self._get_blocks(with_throughput)
    get_num_instructions = self._get_num_instructions_callback(with_throughput)
    batches = tuple(
        training.batches(blocks, get_num_instructions, max_blocks_in_batch=3)
    )
    expected_batches = (blocks[0:3], blocks[3:6], blocks[6:9], [blocks[9]])
    self.assertSequenceEqual(batches, expected_batches)

  @parameterized.named_parameters(
      ('with throughput', True), ('without throughput', False)
  )
  def test_max_instruction_limit(self, with_throughput):
    blocks = self._get_blocks(with_throughput)
    get_num_instructions = self._get_num_instructions_callback(with_throughput)
    # We're using the first 10 blocks from
    # gematria/testing/testdata/basic_blocks_with_throughput.pbtxt.
    # Lengths of blocks in self.blocks are: [1, 5, 1, 8, 3, 4, 1, 2, 9, 4].
    batches = tuple(
        training.batches(
            blocks, get_num_instructions, max_instructions_in_batch=8
        )
    )
    expected_batches = (
        blocks[0:3],
        [blocks[3]],
        blocks[4:7],
        [blocks[7], blocks[9]],
    )
    self.assertSequenceEqual(batches, expected_batches)

  @parameterized.named_parameters(
      ('with throughput', True), ('without throughput', False)
  )
  def test_both_limits(self, with_throughput):
    blocks = self._get_blocks(with_throughput)
    get_num_instructions = self._get_num_instructions_callback(with_throughput)
    batches = tuple(
        training.batches(
            blocks,
            get_num_instructions,
            max_blocks_in_batch=2,
            max_instructions_in_batch=8,
        )
    )
    expected_batches = (
        blocks[0:2],
        [blocks[2]],
        [blocks[3]],
        blocks[4:6],
        blocks[6:8],
        [blocks[9]],
    )
    self.assertSequenceEqual(batches, expected_batches)


class DummyModel(tf.Module):
  """A test model that contains some trainable variables."""

  def __init__(
      self, initial_value: int, var1_spec, var2_spec, var3_spec, var4_spec
  ):
    self.var1 = tf.Variable(
        tf.cast(tf.fill(var1_spec.shape, initial_value), dtype=var1_spec.dtype),
        name='var1',
    )
    self.var2 = tf.Variable(
        tf.cast(tf.fill(var2_spec.shape, initial_value), dtype=var2_spec.dtype),
        name='var2',
    )
    self.var3 = tf.Variable(
        tf.cast(tf.fill(var3_spec.shape, initial_value), dtype=var3_spec.dtype),
        name='var3',
    )
    self.var4 = tf.Variable(
        tf.cast(tf.fill(var4_spec.shape, initial_value), dtype=var4_spec.dtype),
        name='var4',
    )


class PartiallyRestoreFromCheckpointTest(tf.test.TestCase):

  def test_partially_restore(self):
    checkpoint_folder = os.path.join(self.get_temp_dir(), 'checkpoint')
    v1_spec = tf.TensorSpec((3,), dtype=tf.dtypes.int32)

    v2_spec_a = tf.TensorSpec((2, 2), dtype=tf.dtypes.float32)
    v2_spec_b = tf.TensorSpec((2, 2), dtype=tf.dtypes.int32)

    v3_spec_a = tf.TensorSpec((1, 3), dtype=tf.dtypes.float32)
    v3_spec_b = tf.TensorSpec((2, 1), dtype=tf.dtypes.float32)

    v4_spec = tf.TensorSpec((3,), dtype=tf.dtypes.int32)

    model_a = DummyModel(1, v1_spec, v2_spec_a, v3_spec_a, v4_spec)
    checkpoint = tf.train.Checkpoint(model_a)
    model_a_save_path = checkpoint.save(checkpoint_folder)

    model_b = DummyModel(2, v1_spec, v2_spec_b, v3_spec_b, v4_spec)
    training.partially_restore_from_checkpoint(
        model_a_save_path, False, model_b
    )

    self.assertAllEqual(model_b.var1, [1, 1, 1])
    self.assertAllEqual(model_b.var2, [[2, 2], [2, 2]])
    self.assertAllEqual(model_b.var3, [[2], [2]])
    self.assertAllEqual(model_b.var4, [1, 1, 1])


if __name__ == '__main__':
  tf.test.main()
