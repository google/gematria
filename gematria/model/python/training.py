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
"""Contains helper functions and classes for training models."""

from collections.abc import Callable, Iterable, Sequence
import dataclasses
import math
from typing import Optional, TypeVar

from absl import logging
from gematria.basic_block.python import basic_block
from gematria.basic_block.python import throughput
import numpy as np
import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class TrainingEpochStats:
  """Contains statistics from a single step of model training.

  These statistics are meant for printing to log during training and for a human
  reader, a full set of statistics for later analysis is also emitted using the
  TensorFlow summaries APIs.

  The covered statistics include: the selected loss, mean squared errors, mean
  absolute errors, mean absolute percentage errors, and percentiles of absolute
  errors and absolute percentage errors. Statistics that are difficult to
  interpret and not commonly used are not included.

  For models trained using deltas (per-instruction/prefix based expected
  outputs), we collect the stats both for the final output, and for the deltas.
  Note that we do not collect relative errors for deltas: the expected values
  are often very small and relative errors on deltas are thus often ill-defined.

  The loss is stored as a single floating point value; all other values are
  stored per-task, and they are stored as NumPy arrays.

  Attributes:
    epoch: The global step (epoch) at which the stats were collected.
    loss: The loss value used for training the model. In multi-task models, this
      is the combined loss for all tasks.
    percentile_ranks: The list of ranks of the collected percentiles.
    absolute_mse: The absolute mean square error at the step. Has shape
      [num_tasks] where num_tasks is the number of tasks of the model.
    relative_mae: The relative mean absolute error at the step. Has shape
      [num_tasks].
    relative_mse: The relative mean square error at the step. Has shape
      [num_tasks].
    absolute_delta_mse: The absolute mean square error on deltas at the step.
      Only used by seq2seq models. Has shape [num_tasks].
    absolute_delta_mae: The absolute mean error on deltas at the step. Only used
      by seq2seq models. Has shape [num_tasks].
    absolute_error_percentiles: The percentiles of the absolute error. Has shape
      [len(this.percentile_ranks), num_tasks].
    relative_error_percentiles: The percentiles of the relative error. Has shape
      [len(this.percentile_ranks), num_tasks].
    absolute_delta_error_percentiles: The percentiles of the absolute error on
      deltas. Only used by seq2seq models. Has shape
      [len(this.percentile_ranks), num_tasks].
  """

  epoch: tf.Tensor
  loss: float
  percentile_ranks: Sequence[int]
  absolute_mse: tf.Tensor
  relative_mae: tf.Tensor
  relative_mse: tf.Tensor
  absolute_error_percentiles: tf.Tensor
  relative_error_percentiles: tf.Tensor
  absolute_delta_mse: Optional[tf.Tensor] = None
  absolute_delta_mae: Optional[tf.Tensor] = None
  absolute_delta_error_percentiles: Optional[tf.Tensor] = None

  def __post_init__(self) -> None:
    """Validate and finalize the initialization of TrainingEpochStats.

    Raises:
      ValueError: When the sizes of percentile_rank, absolute_error_percentiles,
        and relative_error_percentiles are not aligned.
    """
    # Check that if there are seq2seq stats, none of them is None.
    has_delta_stats = (
        self.absolute_delta_mse is not None
        or self.absolute_delta_mae is not None
        or self.absolute_delta_error_percentiles is not None
    )
    if has_delta_stats and self.absolute_delta_mse is None:
      raise ValueError(
          'Incomplete seq2seq statistics: absolute_delta_mse is missing'
      )
    if has_delta_stats and self.absolute_delta_mae is None:
      raise ValueError(
          'Incomplete seq2seq statistics: absolute_delta_mae is missing'
      )
    if has_delta_stats and self.absolute_delta_error_percentiles is None:
      raise ValueError(
          'Incomplete seq2seq statistics: '
          'absolute_delta_error_percentiles is missing'
      )

    # Check that the dimensions of all the arrays match.
    if len(self.absolute_mse.shape) != 1:
      raise ValueError(
          'Expected absolute_mse to have shape [num_tasks],'
          f' found {self.absolute_mse.shape}'
      )
    num_tasks = self.absolute_mse.shape[0]
    num_percentile_ranks = len(self.percentile_ranks)
    expected_percentile_shape = (num_percentile_ranks, num_tasks)
    if num_percentile_ranks == 0:
      expected_percentile_shape = (0,)

    if self.relative_mae.shape != self.absolute_mse.shape:
      raise ValueError(
          'Expected relative_mae to have shape'
          f' {self.absolute_mse.shape}, found'
          f' {self.relative_mae.shape}'
      )
    if self.relative_mse.shape != self.absolute_mse.shape:
      raise ValueError(
          'Expected relative_mse to have shape'
          f' {self.absolute_mse.shape}, found'
          f' {self.relative_mae.shape}'
      )

    if self.absolute_error_percentiles.shape != expected_percentile_shape:
      raise ValueError(
          'Expected absolute_error_percentiles to have shape'
          f' [{num_percentile_ranks}, {num_tasks}], found'
          f' {self.absolute_error_percentiles.shape}'
      )
    if self.relative_error_percentiles.shape != expected_percentile_shape:
      raise ValueError(
          'Expected relative_error_percentiles to have shape'
          f' [{num_percentile_ranks}, {num_tasks}], found'
          f' {self.absolute_error_percentiles.shape}'
      )

    if has_delta_stats:
      # We already checked that all of them are present above.
      if self.absolute_delta_mse.shape != self.absolute_mse.shape:
        raise ValueError(
            'Expected absolute_delta_mse to have shape'
            f' {self.absolute_mse.shape}, found'
            f' {self.absolute_delta_mse.shape}'
        )
      if self.absolute_delta_mae.shape != self.absolute_mse.shape:
        raise ValueError(
            'Expected absolute_delta_mae to have shape'
            f' {self.absolute_mse.shape}, found'
            f' {self.absolute_delta_mae.shape}'
        )
      if (
          self.absolute_delta_error_percentiles.shape
          != expected_percentile_shape
      ):
        raise ValueError(
            'Expected absolute_delta_error_percentiles to have shape'
            f' {self.relative_error_percentiles.shape}, found'
            f' {self.absolute_delta_error_percentiles.shape}'
        )

  def __str__(self) -> str:
    """Converts the stats to a human-readable string."""
    parts = [
        f'epoch: {_unwrap_if_tensor(self.epoch)}, loss: {self.loss}',
        self._format_loss_string(
            'absolute',
            _unwrap_if_tensor(self.absolute_mse),
            None,
            _unwrap_if_tensor(self.absolute_error_percentiles),
        ),
        self._format_loss_string(
            'relative',
            _unwrap_if_tensor(self.relative_mse),
            _unwrap_if_tensor(self.relative_mae),
            _unwrap_if_tensor(self.relative_error_percentiles),
        ),
    ]
    if self.absolute_delta_mse is not None:
      parts.append(
          self._format_loss_string(
              'absolute delta',
              _unwrap_if_tensor(self.absolute_delta_mse),
              _unwrap_if_tensor(self.absolute_delta_mae),
              _unwrap_if_tensor(self.absolute_delta_error_percentiles),
          )
      )
    return '\n'.join(parts)

  def _format_loss_string(
      self,
      name: str,
      mse: Sequence[float],
      mae: Optional[Sequence[float]],
      percentiles: Sequence[Sequence[float]],
  ) -> str:
    """Returns a human-readable loss information for use in logs."""
    parts = [f'{name} mse: {_as_list(mse)}']
    if mae is not None:
      parts.append(f', mae: {_as_list(mae)}')
    for i, rank in enumerate(self.percentile_ranks):
      parts.append(f', {rank}%: {_as_list(percentiles[i])}')
    return ''.join(parts)


T = TypeVar('T')


def get_num_instructions_in_block(block: basic_block.BasicBlock) -> int:
  """Returns the number of instructions in a basic block."""
  return len(block.instructions)


def get_num_instructions_in_block_with_throughput(
    block: throughput.BasicBlockWithThroughput,
) -> int:
  """Returns the number of instructions in a basic block with throughput."""
  return len(block.block.instructions)


def batches(
    blocks: Iterable[T],
    get_num_instructions: Callable[[T], int],
    max_blocks_in_batch: Optional[int] = None,
    max_instructions_in_batch: Optional[int] = None,
) -> Iterable[Sequence[T]]:
  """Splits 'blocks' into a sequence of batches respecting the size limits.

  When max_blocks_in_batch is specified, each batch has at most this number of
  basic blocks in it. When max_instructions_in_batch is specified, each batch
  has at most this number of instructions in it across all basic blocks; blocks
  that have more instructions than max_instructions_in_batch are skipped.

  The order of the basic blocks in the batches is the same as in the input
  sequence.

  For example, suppose that block(n) returns a basic block with n instructions,
  and blocks = [block(n) for n in (1, 3, 5, 1, 15, 12)].
  Then
    Batches(blocks, max_blocks_in_batch=3, max_instructions_in_batch = 12)
  returns [[blocks[0:3], [blocks[3]], [blocks[5]]].

  Args:
    blocks: The basic block collection that is split into batches.
    get_num_instructions: A callback that returns the number of instructions in
      each basic block.
    max_blocks_in_batch: The number of basic blocks to include in a single
      batch. When not specified, the number of basic blocks per batch is not
      limited.
    max_instructions_in_batch: The maximal number of instructions in a single
      batch. When not specified, the number of instructions per batch is not
      limited.

  Yields:
    The basic blocks from the input sequence, in the original order, split into
    batches following the specified limits.
  """
  max_instructions_in_batch = max_instructions_in_batch or math.inf
  max_blocks_in_batch = max_blocks_in_batch or math.inf

  current_batch = []
  num_instructions_in_batch = 0
  for block in blocks:
    num_instructions_in_block = get_num_instructions(block)
    if num_instructions_in_block > max_instructions_in_batch:
      # This block alone has more instruction than we allow in a single batch.
      # We skip this basic block.
      logging.warn(
          (
              'Single basic block has more instructions (%d) than the '
              'allowed limit per batch (%d). Skipping the basic block'
          ),
          num_instructions_in_block,
          max_instructions_in_batch,
      )
      continue
    new_instructions_in_batch = (
        num_instructions_in_batch + num_instructions_in_block
    )
    if (
        new_instructions_in_batch > max_instructions_in_batch
        or len(current_batch) == max_blocks_in_batch
    ):
      yield current_batch
      current_batch = []
      num_instructions_in_batch = 0
    current_batch.append(block)
    num_instructions_in_batch += num_instructions_in_block
  if current_batch:
    yield current_batch


def partially_restore_from_checkpoint(
    checkpoint_file: str, load_step_from_ckpt: bool, model: tf.Module
) -> None:
  """Partially restores a checkpoint to the current graph.

  Reads the list of variables from a checkpoint and from the current graph, and
  restores the values of all variables in the graph that have compatible data in
  the checkpoint, i.e. the checkpoint contains a tensor with a matching name
  that has a compatible shape and the same dtype.

  Args:
    checkpoint_file: A checkpoint to partially restore from.
    load_step_from_ckpt: If True, load the step value from the given checkpoint
      file.
    model: The tf.Module object representing the model that the weights should
      be restored into.
  """
  reader = tf.train.load_checkpoint(checkpoint_file)
  shapes = reader.get_variable_to_shape_map()
  dtypes = reader.get_variable_to_dtype_map()

  if load_step_from_ckpt:
    tf.compat.v1.assign(
        tf.compat.v1.train.get_or_create_global_step(),
        reader.get_tensor('global_step'),
    )

  for variable in model.trainable_variables:
    # All variable names should end with ':0'; this ':0' is not used in the
    # checkpoint.
    if not variable.name.endswith(':0'):
      continue
    variable_name = variable.name[:-2] + '/.ATTRIBUTES/VARIABLE_VALUE'
    if variable_name not in shapes:
      logging.info('%s not found in the checkpoint', variable_name)
      continue
    if not variable.shape.is_compatible_with(shapes[variable_name]):
      logging.info(
          '%s does not have the right shape:\n\tCheckpoint: %r\n\tGraph: %r',
          variable_name,
          shapes[variable_name],
          variable.shape,
      )
      continue
    if variable.dtype.base_dtype != dtypes[variable_name].base_dtype:
      logging.info(
          '%s does not have the right dtype:\n\tCheckpoint: %r\n\tGraph: %r',
          variable_name,
          dtypes[variable_name],
          variable.dtype,
      )
      continue
    logging.info('Restoring %s', variable_name)
    variable.load(reader.get_tensor(variable_name))


def _as_list(values: Sequence[float]) -> list[float]:
  """Converts `values` to a list."""
  if isinstance(values, np.ndarray):
    return values.tolist()
  return list(values)


def _unwrap_if_tensor(
    maybe_tensor: tf.Tensor | Sequence[float] | float,
) -> Sequence[float] | float:
  """Unwraps `maybe_tensor` if it is a Tensor."""
  if tf.is_tensor(maybe_tensor):
    return maybe_tensor.numpy()
  return maybe_tensor
