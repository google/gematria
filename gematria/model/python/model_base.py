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
"""Defines ModelBase, a base class for Gematria models.

The base class provides the necessary tools for building Gematria models:
 * A framework for defining the models.
 * Loss computation, summaries and optimizers based on the loss.
 * Training loop supporting random mini-batches.
 * Model evaluation loop.
 * Inference functionality.

Concrete models can be derived either directly from this class, or by inheriting
from classes that extend ModelBase with support for a particular library, e.g.
GnnModelBase.
"""

import abc
import collections
from collections.abc import Iterable, MutableMapping, MutableSequence, Sequence, Callable
import itertools
import math
import os
import random
from typing import Optional, TypeVar, Union, TypedDict

from absl import logging
from gematria.basic_block.python import basic_block
from gematria.basic_block.python import throughput
from gematria.model.python import loss_utils
from gematria.model.python import options
from gematria.model.python import training
from gematria.utils.python import timer
import numpy as np
import scipy.stats
import tensorflow as tf
import tf_slim.evaluation

# The type used for TensorFlow feed_dict objects. The type we use here is
# simpler than what is actually accepted by TensorFlow, but the typing should be
# sufficient for our use. Moreover, since TensorFlow and NumPy do not provide
# type annotations, both the key and the value are reduced to typing.Any.
FeedDict = MutableMapping[str, Union[np.ndarray, tf.Tensor]]

# A throughput value used as a placeholder in the expected output tensors for
# masked expected outputs.
INVALID_THROUGHPUT_VALUE = -1

_BASIC_BLOCK_INDEX_TF_DTYPE = tf.dtypes.int32
_BASIC_BLOCK_INDEX_NUMPY_DTYPE = _BASIC_BLOCK_INDEX_TF_DTYPE.as_numpy_dtype()

# A type variable that is either a basic block or block with throughput. The
# advantage over typing.Union is that in each context, the typevar represents
# only one of the base types, and e.g. Sequence[BlockOrBlockWithThroughput] is
# a sequence of blocks or a sequence of blocks with throughput, but it can't be
# a sequence that contains both types.
BlockOrBlockWithThroughput = TypeVar(
    'BlockOrBlockWithThroughput',
    basic_block.BasicBlock,
    throughput.BasicBlockWithThroughput,
)


class AddBasicBlockError(Exception):
  """The exception raised when adding a block to batch fails."""


class SaveBestCheckpoint(tf.compat.v1.train.SessionRunHook):
  """A run hook that saves top N models based on error values."""

  def __init__(
      self,
      error_tensor: tf.Tensor,
      checkpoint_dir: str,
      global_step: int,
      max_to_keep: int = 15,
  ):
    """Initializes the hook.

    Args:
      error_tensor: The tensor of errors, used to pick the best models.
      checkpoint_dir: The model checkpoint directory.
      global_step: The model global step.
      max_to_keep: The number of best checkpoints to keep.
    """
    self._error_tensor = error_tensor
    self._checkpoint_dir = checkpoint_dir
    self._global_step = global_step

    self._saver = tf.train.Saver(max_to_keep=max_to_keep, name='relative_mae')
    self._last_eval = math.inf

  def before_run(self, run_context: ...) -> tf.compat.v1.train.SessionRunArgs:
    return tf.compat.v1.train.SessionRunArgs(
        {'loss': self._error_tensor, 'global_step': self._global_step}
    )

  def after_run(self, run_context: ..., run_values: ...):
    step = run_values.results['global_step']
    value = run_values.results['loss']
    # With multiple tasks, we'd get an array of bools which does not have a well
    # defined boolean conversion. We save the checkpoint if it is better on any
    # of the tasks.
    if (value < self._last_eval).any():
      self._last_eval = np.minimum(value, self._last_eval)
      self._saver.save(
          sess=run_context.session,
          save_path=os.path.join(self._checkpoint_dir, 'best_models'),
          global_step=step,
      )


class OutputDict(TypedDict, total=False):
  """Represents the outputs of executing the model.

  This class is designed to be used for typing the outputs of executing a
  model in all of its various modes.
  """

  output: tf.Tensor
  output_deltas: tf.Tensor
  output_mask_deltas: tf.Tensor


class ModelBase(tf.Module, metaclass=abc.ABCMeta):
  """Base class for Gematria basic block processing models.

  Provides infrastructure for building basic-block-oriented models on top of
  TensorFlow:
  - a framework for implementing models, while reusing common functionality,
  - API for creating batches of a given size out of the input data,
  - API for training and evaluation of the model,
  - API for providing inference as a service with the model.

  A concrete model is typically implemented by overriding the following
  ModelBase methods:
  - _create_tf_graph() to add model-specific TensorFlow ops,
  - _add_basic_block_to_batch() to implement model-specific basic block
    traversal,
  - _make_batch_feed_dict() to export a TensorFlow feed_dict from the current
    batch,
  - _MakeSourceName() to provide a default hyperparameter-based name for
    ThroughputWithSourceProto,
  - optionally overriding other methods to further customize the behavior of the
    class.

  Usually, models would not be based on this class directly, but on child
  classes that add support for a particular ML framework, such as GnnModelBase.

  Attributes:
    _expected_outputs: the expected output tensor.
    _loss_type: The type of loss used by the model.
    _loss_normalization: Specifies whether and how the errors are normalized in
      the loss value used for training the model.
    _collected_percentile_ranks: The ranks of the percentiles of absolute and
      relative error that are collected by the model. Contains an empty list
      when no percentiles are collected.
    _loss_tensor: The loss tensor used for training. The tensor contains the
      loss created as specified by _loss_type and _loss_normalization.
    _optimizer: the TensorFlow optimizer used to minimize the loss. We use
      AdamOptimizer.
    _train_step: the op that runs the training step for the optimizer when
      evaluated.
    _is_chief: When running in a distributed training setup, the property is set
      to True if this is the model for the chief worker.
    _synchronous_training: When running in a distributed training setup, the
      property is set to True if the training uses synchronous weight updates.
    _num_training_worker_replicas: The total number of training worker replicas
      in a distributed training setup.
    _num_training_worker_replicas_to_aggregate: When running distributed
      training with synchronous weight updats, this is the (minimal) number of
      replicas that is aggregated into a single step of the training. Keeping
      this number lower than the total number of replicas makes the training
      faster as it doesn't have to wait for the slowest replica.
  """

  # Error message used when the model was created with use_deltas=False, and the
  # user tries to access a property that is valid only for models created with
  # use_deltas=True. This is a string template, that takes the name of the
  # property as its only argument.
  _USE_DELTAS_ATTRIBUTE_ERROR_MESSAGE = (
      '%s is available only when the model is created in the sequence to'
      ' sequence mode (use_deltas is True)'
  )

  # The name used for the tensor that contains the per-basic block outputs of
  # the model (stored in self._output_tensor). This name is used both in the
  # seq2num and the seq2seq models.
  #  * In the seq2num mode, ModelBase._create_output_and_loss_tensors checks
  #    that the output tensor has the right name and wraps it in a tf.identity()
  #    op changing the name if needed.
  #  * In the seq2seq mode, self._output_tensor is created by the framework from
  #    self._output_tensor_deltas with the correct name.
  OUTPUT_TENSOR_NAME = 'ModelBase.output_tensor'
  # The name of the tensor that contains the per-instruction outputs of the
  # model. This name is used in the seq2seq mode.
  # ModelBase._create_output_and_loss_tensors checks that
  # self._output_tensor_deltas has the right name and wraps it with a
  # tf.identity() op changing the name if needed.
  OUTPUT_TENSOR_DELTAS_NAME = 'ModelBase.output_tensor_deltas'

  def __init__(
      self,
      *,
      dtype: tf.dtypes.DType,
      loss_type: options.LossType = options.LossType.MEAN_SQUARED_ERROR,
      loss_normalization: options.ErrorNormalization = (
          options.ErrorNormalization.NONE
      ),
      use_deltas: bool = False,
      use_delta_loss: bool = True,
      create_delta_block_index: bool = False,
      optimizer_type: Optional[
          options.OptimizerType
      ] = options.OptimizerType.ADAM,
      grad_clip_norm: Optional[float] = None,
      learning_rate: Optional[float] = 0.001,
      decay_steps: Optional[int] = 0,
      decay_rate: Optional[float] = 0.0,
      learning_rate_schedule: Optional[options.LearningRateScheduleType] = (
          options.LearningRateScheduleType.NONE
      ),
      collected_percentile_ranks: Optional[Sequence[int]] = None,
      synchronous_training: bool = False,
      is_chief: bool = False,
      num_training_worker_replicas: int = 1,
      num_training_worker_replicas_to_aggregate: int = 1,
      model_name: Optional[str] = None,
      task_list: Optional[Sequence[str]] = None,
      trained_variable_groups: Optional[Iterable[str]] = None,
  ) -> None:
    """Creates a new model with the provided parameters.

    The constructor initializes the model object, but it does not create any
    TensorFlow objects yet. This must be done by calling Initialize() on the
    newly created model.

    Args:
      dtype: The TensorFlow DType of the output of the model.
      loss_type: The loss type used by the model.
      loss_normalization: The type of error normalization used when computing
        the loss used to train the model.
      use_deltas: Return sequence of predictions for each instruction.
      use_delta_loss: Determines how loss is computed in the sequence prediction
        mode. When True, the error is computed for each element of the sequence
        separately, which is compared to the difference between prefix
        throughputs corresponding to the instruction. When False, the error is
        computed for each basic block once as the difference of the sum of the
        output values and the expected throughput of the whole block.
      create_delta_block_index: When True, the model creates the delta block
        index tensor even when running in the seq2num mode.
      optimizer_type: Specifies a TF optimizer type for training.
      grad_clip_norm: Specifies to clipping value for gradient. If None, no
        clipping will be applied.
      learning_rate: The learning rate passed to the optimizer when creating the
        model. When None, the default learning rate of the model is used.
      decay_steps: Depending on the learning schedule type, determines the
        maximum number of decay steps.
      decay_rate: Depending on the learning schedule type, determines the rate
        at which the learning rate is decayed. Note that some learning rate
        schedules do not use `decay_rate`.
      learning_rate_schedule: Determines the type of learning rate schedule. To
        enable learning rate decay, also set `decay_steps` > 0 and
        `learning_rate_schedule` != options.LearningRateScheduleType.NONE.
      collected_percentile_ranks: The list of percentile ranks of the errors
        collected during training and evaluation. The percentiles are exported
        through TensorFlow summaries and printed to log. By default, no error
        percentiles are collected.
      synchronous_training: Use synchronous weight updates when running in a
        distributed training setup.
      is_chief: Create a model for the chief training worker when running in a
        distributed training setup.
      num_training_worker_replicas: The total number of training worker replicas
        used in the distributed training setup.
      num_training_worker_replicas_to_aggregate: When running distributed
        training with synchronous weight updats, this is the (minimal) number of
        replicas that is aggregated into a single step of the training. Keeping
        this number lower than the total number of replicas makes the training
        faster as it doesn't have to wait for the slowest replica.
      model_name: The name of the model used in inference mode; this name is
        used in the output BasicBlockWithThroughputProtos. When not specified, a
        default name based on the model class and its hyperparameters is used.
      task_list: A list of task names. Has to have the same task ordering as in
        the data (basic_block.inverse_througputs field). When it is not
        provided, the model initializes with a single task called "default".
      trained_variable_groups: The list of variable group names that are trained
        by the optimizer. Only variables in this list are touched; if the list
        is empty or None, all variables are trained.
    """
    self._dtype = dtype
    self._numpy_dtype = dtype.as_numpy_dtype
    self._loss_normalization = loss_normalization
    self._use_deltas = use_deltas
    self._use_delta_loss = use_delta_loss
    self._learning_rate = learning_rate
    self._decay_steps = decay_steps
    self._decay_rate = decay_rate
    self._learning_rate_schedule = learning_rate_schedule
    self._loss_type = loss_type
    self._collected_percentile_ranks = collected_percentile_ranks or []
    self._is_chief = is_chief
    self._synchronous_training = synchronous_training
    self._num_training_worker_replicas = num_training_worker_replicas
    self._num_training_worker_replicas_to_aggregate = (
        num_training_worker_replicas_to_aggregate
    )
    self._optimizer_type = optimizer_type
    self._grad_clip_norm = grad_clip_norm
    task_list = task_list or ('default',)
    self._task_list: Sequence[str] = task_list

    self._model_name = model_name

    # Named groups of variable tensors in the model. The training procedure
    # allows restricting weight updates only to variables from a certain group.
    self._variable_groups: MutableMapping[str, MutableSequence[tf.Tensor]] = (
        collections.defaultdict(list)
    )
    self._trained_variable_groups = tuple(trained_variable_groups or ())

    self._global_step = tf.compat.v1.train.get_or_create_global_step()

    self._decayed_learning_rate = None
    self._loss: Optional[loss_utils.LossComputation] = None
    self._delta_loss: Optional[loss_utils.LossComputation] = None
    self._train_step: Optional[tf.Operation] = None
    self._optimizer: Union[
        tf.train.Optimizer, tf.train.SyncReplicasOptimizer
    ] = None
    # Specifies whether self._delta_block_index_tensor should be created. This
    # property is always True for seq2seq models, but it can be also requested
    # by seq2num models that need to partition per-instruction data in the batch
    # into basic blocks.
    self._create_delta_block_index = create_delta_block_index or use_deltas
    # Used only when self._create_delta_block_index is True. This tensor is
    # parallel with the self._output_tensor_deltas, and
    # self._delta_block_index_tensor[i] contains the index of the basic block
    # that owns self._output_tensor_deltas[i].
    # TODO(ondrasej): Consider removing the 'delta' from the name, since this
    # might be used also outside without the seq2seq mode.
    self._delta_block_index_tensor: Optional[tf.Tensor] = None

    # The following fields are used when scheduling a batch. They are
    # initialized by ModelBase._start_batch() and cleaned up by
    # ModelBase._finalize_batch(). They are set to None otherwise.

    # The list of expected output values for the basic blocks in the batch, in
    # the order in which they were added to the batch. Converted to a NumPy
    # array, the final version can be used as a value for
    # self._expected_outputs.
    self._batch_expected_outputs: Optional[
        MutableSequence[MutableSequence[int | float]]
    ] = None
    # The list of expected values for basic block prefixes (deltas). The values
    # are stored in a single list, in the order in which the basic blocks were
    # added to the batch. Within each basic block, the predictions for the
    # deltas in the order in which the corresponding instructions appear in the
    # basic block.
    self._batch_expected_outputs_deltas: Optional[MutableSequence[float]] = None
    # The list of basic block indices corresponding to the deltas. This list is
    # used during the scheduling of a batch to collect the values that are then
    # fed into self._delta_basic_block_tensor via the feed_dict.
    self._batch_delta_block_index: Optional[MutableSequence[int]] = None
    # The list of per-basic block masks of valid expected outputs. Each element
    # of the list is a sequence of length self.num_tasks, where
    # self._batch_mask[block][task] is True iff the expected output of the
    # given task for the given block is valid.
    self._batch_mask: Optional[MutableSequence[Sequence[bool]]] = None

  def initialize(self) -> None:
    """Initializes the model. Must be called before any other method."""
    self._create_optimizer()

  @property
  def use_deltas(self) -> bool:
    """Returns True for sequence to sequence models."""
    return self._use_deltas

  @property
  def dtype(self) -> tf.dtypes.DType:
    """Returns the TensorFlow dtype used in the model."""
    return self._dtype

  @property
  def numpy_dtype(self) -> np.dtype:
    """Returns the NumPy dtype used in the model.

    The this is the NumPy equivalent of self.dtype() for use with NumPy APIs.
    """
    return self._numpy_dtype

  @property
  def num_tasks(self) -> int:
    """Returns the number of tasks in the model."""
    return len(self._task_list)

  @property
  def task_list(self) -> Sequence[str]:
    """Returns the names of the tasks in the model."""
    return self._task_list

  @property
  def loss_type(self) -> options.LossType:
    """Returns the type of loss used by the model."""
    return self._loss_type

  @property
  def loss_normalization(self) -> options.ErrorNormalization:
    """Returns the error normalization used when computing the loss."""
    return self._loss_normalization

  @property
  def collected_percentile_ranks(self) -> Sequence[int]:
    """Returns the list of collected error percentile ranks."""
    return self._collected_percentile_ranks

  @property
  def global_step(self) -> tf.Tensor:
    """Returns the global step tensor for the model."""
    return self._global_step

  @property
  def model_name(self) -> str:
    """Returns the name of the model.

    Returns the name provided during the construction of the model, if one was
    provided. Otherwise, returns a name based on the model class and its
    parameters.
    """
    return self._model_name or self._make_model_name()

  @property
  def output_tensor_names(self) -> Sequence[str]:
    """Returns the list of names of the output tensors of the models.

    This property is used when exporting the TensorFlow graph of a trained model
    into a frozen graph def. Child classes can override this method to change
    the output tensors or add additional tensors to the list.
    """
    if self.use_deltas:
      return (ModelBase.OUTPUT_TENSOR_NAME, ModelBase.OUTPUT_TENSOR_DELTAS_NAME)
    return (ModelBase.OUTPUT_TENSOR_NAME,)

  @abc.abstractmethod
  def _forward(self, feed_dict: FeedDict) -> OutputDict:
    """Implements the forward pass of the model.

    This function should be implemented in downstream models and calculate the
    outputs of the model given the inputs specified in feed_dict.

    Returns:
      A dictionary containing tensors. This should contain a key, 'output' in
      seq2num mode and a key named 'output_deltas' in seq2seq mode.
    """
    pass

  def __call__(self, feed_dict: FeedDict, train=False) -> OutputDict:
    """Implements the non-model specific part of the forward pass.

    This function wraps the _forward method and does relevant calculations
    when working with models that use deltas.
    """
    if not self._use_deltas:
      return self._forward(feed_dict)

    output = self._forward(feed_dict)

    if train:
      output['output_mask_deltas'] = tf.nn.embedding_lookup(
          feed_dict['output_mask'],
          feed_dict['delta_block_index'],
          name='ModelBase.output_mask_deltas',
      )

    output['output'] = tf.math.segment_sum(
        output['output_deltas'],
        feed_dict['delta_block_index'],
        name=ModelBase.OUTPUT_TENSOR_NAME,
    )

    return output

  @abc.abstractmethod
  def _make_model_name(self) -> str:
    """Returns a model name based on its class and parameters."""

  def get_source_name(self, task_index: int) -> str:
    """Returns the source name for ThroughputWithSourceProto for this model."""
    if task_index < 0 or task_index >= self.num_tasks:
      raise ValueError(f'Invalid task index: {task_index}')
    task_name = self.task_list[task_index]
    return f'{self.model_name}, task={task_name}'

  def _add_percentile_summaries(
      self,
      error_name: str,
      percentile_ranks: Sequence[int],
      percentile_tensor: tf.Tensor,
  ) -> None:
    """Adds TF percentile summaries with task names.

    Args:
      error_name: The name of the error, used for naming the summaries.
      percentile_ranks: The list of percentile ranks for which the summaries are
        added.
      percentile_tensor: The tensor that contains the values of the percentiles.
        The tensor must have the same number of elements as percentile_ranks.
    """
    expected_percentile_tensor_shape = (len(percentile_ranks), self.num_tasks)
    if (
        percentile_ranks
        and percentile_tensor.shape != expected_percentile_tensor_shape
    ):
      raise ValueError(
          f'The number of percentile ranks ({len(percentile_ranks)}) and tasks '
          f'({self.num_tasks}) does not match the shape of the tensor '
          f'({percentile_tensor.shape}).'
      )
    for task_idx, task_name in enumerate(self._task_list):
      for percentile_idx, rank in enumerate(percentile_ranks):
        # TODO(ondrasej): Maybe we could simplify this as
        # percentile = percentile_tensor[percentile_idx, task_idx].
        percentile = tf.gather(percentile_tensor[:, task_idx], percentile_idx)
        tf.summary.scalar(f'{error_name}_{rank}p_{task_name}', percentile)

  def _add_error_summaries(self, error_name: str, error_tensor: tf.Tensor):
    if error_tensor.shape[0] != self.num_tasks:
      raise ValueError(
          'First dimension of thr error tensor should be equal to'
          f' the number of tasks: {self.num_tasks}.'
      )
    for task_idx, task_name in enumerate(self._task_list):
      summary_name = f'{error_name}_{task_name}'
      tf.summary.scalar(summary_name, error_tensor[task_idx])

  def _make_spearman_correlations(
      self, expected_outputs: tf.Tensor, output_tensor: tf.Tensor
  ) -> tf.Tensor:
    """Creates a contains Spearman rank correlation tensor.

    Assumes that both inputs have rank (None, num_tasks). Returns a tensor in
    the shape (num_tasks,) that contains the Spearman rank correlation
    coefficients between the corresponding columns of `expected_outputs` and
    `output_tensor`.

    Args:
      expected_outputs: The expected output values.
      output_tensor: The output values computed by the model.

    Returns:
      A tensor that contains the Spearman rank correlations between columns of
      the input tensors.
    """
    task_correlations = []
    # The Spearman rank correlation from scipy can take 2D inputs, but in such
    # case it computes a correlation matrix between all columns of the both
    # inputs. To compute only the coefficients we're interested in, we need to
    # do for each task separately.
    for task in range(self.num_tasks):
      task_expected_outputs = expected_outputs[:, task]
      task_outputs = output_tensor[:, task]
      task_correlations.append(
          tf.reshape(
              tf.py_function(
                  scipy.stats.spearmanr,
                  (task_expected_outputs, task_outputs),
                  Tout=self.dtype,
              ),
              (1,),
          )
      )
    return tf.concat(task_correlations, axis=0)

  def _clip_if_not_none(self, grad: Optional[tf.Tensor]) -> tf.Tensor:
    if grad is None:
      return grad
    return tf.clip_by_norm(grad, self._grad_clip_norm)

  def _create_optimizer(self) -> None:
    """Creates an optimizer for the model."""
    decay_args = {
        'learning_rate': self._learning_rate,
        'global_step': self.global_step,
        'decay_steps': self._decay_steps,
    }
    decay_rate_arg = {'decay_rate': self._decay_rate}

    if (
        self._learning_rate_schedule != options.LearningRateScheduleType.NONE
    ) and (self._decay_steps == 0):
      raise ValueError(
          'When a learning schedule is selected, `decay_steps` '
          'must be great than zero.'
      )
    if self._learning_rate_schedule == options.LearningRateScheduleType.COSINE:
      self._decayed_learning_rate = tf.compat.v1.train.cosine_decay(
          **decay_args
      )
    elif (
        self._learning_rate_schedule
        == options.LearningRateScheduleType.EXPONENTIAL
    ):
      self._decayed_learning_rate = tf.compat.v1.train.exponential_decay(
          **decay_args, **decay_rate_arg
      )
    elif (
        self._learning_rate_schedule
        == options.LearningRateScheduleType.INVERSE_TIME
    ):
      self._decayed_learning_rate = tf.compat.v1.train.inverse_time_decay(
          **decay_args, **decay_rate_arg
      )
    elif (
        self._learning_rate_schedule
        == options.LearningRateScheduleType.LINEAR_COSINE
    ):
      self._decayed_learning_rate = tf.compat.v1.train.linear_cosine_decay(
          **decay_args
      )
    elif (
        self._learning_rate_schedule
        == options.LearningRateScheduleType.NATURAL_EXP
    ):
      self._decayed_learning_rate = tf.compat.v1.train.natural_exp_decay(
          **decay_args, **decay_rate_arg
      )
    elif (
        self._learning_rate_schedule
        == options.LearningRateScheduleType.NOISY_LINEAR_COSINE
    ):
      self._decayed_learning_rate = (
          tf.compat.v1.train.noisy_linear_cosine_decay(**decay_args)
      )
    elif (
        self._learning_rate_schedule
        == options.LearningRateScheduleType.POLYNOMIAL
    ):
      self._decayed_learning_rate = tf.compat.v1.train.polynomial_decay(
          **decay_args
      )
    elif (
        self._learning_rate_schedule
        == options.LearningRateScheduleType.COSINE_RESTARTS
    ):
      decay_args['first_decay_steps'] = decay_args.pop('decay_steps')
      self._decayed_learning_rate = tf.train.cosine_decay_restarts(**decay_args)

    else:
      assert (
          self._learning_rate_schedule == options.LearningRateScheduleType.NONE
      )
      self._decayed_learning_rate = self._learning_rate

    if self._optimizer_type == options.OptimizerType.ADAM:
      self._optimizer = tf.compat.v1.train.AdamOptimizer(
          learning_rate=self._decayed_learning_rate
      )
    elif self._optimizer_type == options.OptimizerType.SGD:
      self._optimizer = tf.compat.v1.train.GradientDescentOptimizer(
          learning_rate=self._decayed_learning_rate
      )
    elif self._optimizer_type == options.OptimizerType.RMSPROP:
      self._optimizer = tf.compat.v1.train.RMSPropOptimizer(
          learning_rate=self._decayed_learning_rate
      )
    else:
      raise ValueError(
          'Optimizer %s is not supported. List of supported optimizers are %s'
          % (self._optimizer_type, list(map(str, options.OptimizerType)))
      )
    if self._synchronous_training and self._num_training_worker_replicas > 1:
      # TODO(ondrasej): Rewrite this using the tf.distributed framework.
      # SyncReplicasOptimizer is deprecated and might be removed in the future.
      self._optimizer = tf.train.SyncReplicasOptimizer(
          self._optimizer,
          replicas_to_aggregate=self._num_training_worker_replicas_to_aggregate,
          total_num_replicas=self._num_training_worker_replicas,
      )
    elif self._synchronous_training:
      logging.warning(
          'ModelBase._synchronous_training is True with a single worker.'
      )

  def get_monitored_training_session_hooks(
      self,
  ) -> Sequence[tf.compat.v1.train.SessionRunHook]:
    """Returns hooks for a MonitoredTrainingSession required by the model."""
    hooks = []
    if isinstance(self._optimizer, tf.train.SyncReplicasOptimizer):
      hooks.append(self._optimizer.make_session_run_hook(self._is_chief))
    return hooks

  def validate_basic_block(self, block: basic_block.BasicBlock) -> bool:
    """Checks that a basic block can pre processed by the model.

    By default, all blocks are accepted. Models can override this method to
    reject basic blocks that are not accepted.

    Args:
      block: The basic block to check.

    Returns:
      True if the basic block can be processed by the model; False, otherwise.
    """
    del block  # Unused.
    return True

  def validate_basic_block_with_throughput(
      self, block: throughput.BasicBlockWithThroughput
  ) -> bool:
    """A version of validate_basic_block that works on blocks with throughpu."""
    return self.validate_basic_block(block.block)

  def _start_batch(self) -> None:
    """Method called before adding basic blocks in self.schedule_batch().

    Initializes the resources needed to schedule a new batch. Child classes can
    override this method to initialize model-specific resources.
    """
    assert self._batch_expected_outputs is None, (
        'ModelBase._start_batch() was called without calling '
        'ModelBase._finalize_batch() for the previous batch.'
    )

    self._batch_expected_outputs = []
    self._batch_expected_outputs_deltas = []
    self._batch_delta_block_index = []
    self._batch_mask = []

  @abc.abstractmethod
  def _make_batch_feed_dict(self) -> FeedDict:
    """Creates a feed_dict for the model-specific tensors in the current batch.

    The returned feed_dict object should contain assignments for placeholders
    created by the model. Assignments for placeholders created by ModelBase
    (e.g. self._expected_outputs) are added in ModelBase._finalize_batch().

    Returns:
      A feed_dict object that can be used for running the batch.
    """
    raise NotImplementedError(
        'ModelBase._make_batch_feed_dict() is not implemented'
    )

  def _finalize_batch(self, include_expected_outputs: bool) -> FeedDict:
    """Method called after traversing the all basic blocks in the batch.

    This method is called at the end of self.schedule_batch(). It creates the
    feed_dict object for running the batch in the model, and cleans up resources
    used during scheduling of the batch.

    Args:
      include_expected_outputs: Determines whether the returned feed_dict should
        include values for self._expected_outputs (and
        self._expected_outputs_deltas, when applicable).

    Returns:
      The feed_dict object that can be passed to tf.Session().run() to run the
      current batch.
    """
    schedule = self._make_batch_feed_dict()
    if self._create_delta_block_index:
      schedule['delta_block_index'] = np.array(
          self._batch_delta_block_index, dtype=_BASIC_BLOCK_INDEX_NUMPY_DTYPE
      )
    if include_expected_outputs:
      schedule['expected_outputs'] = np.reshape(
          np.array(self._batch_expected_outputs, dtype=self.numpy_dtype),
          [-1, self.num_tasks],
      )
      schedule['output_mask'] = np.array(self._batch_mask, dtype=bool)
      if self._use_deltas:
        schedule['expected_outputs_deltas'] = np.reshape(
            np.array(
                self._batch_expected_outputs_deltas, dtype=self.numpy_dtype
            ),
            [-1, self.num_tasks],
        )

    # Clean up the batch resources.
    self._batch_expected_outputs = None
    self._batch_expected_outputs_deltas = None
    self._batch_delta_block_index = None

    return schedule

  @abc.abstractmethod
  def _add_basic_block_to_batch(self, block: basic_block.BasicBlock) -> None:
    """Adds a basic block to the current batch.

    Runs the model-specific code for scheduling the batch and updates all data
    structures with the results.

    Args:
      block: The basic block added to the batch.
    """
    raise NotImplementedError(
        'ModelBase._add_basic_block_to_batch is not implemented'
    )

  def _add_expected_outputs_to_batch(
      self,
      throughputs: Sequence[Optional[throughput.BasicBlockThroughput]],
      randomize_expected_outputs: bool,
      num_prefixes: Optional[int] = None,
  ) -> None:
    """Adds expected outputs for a basic block to the batch.

    Args:
      throughputs: The throughput information for a single basic block. The list
        must have a size greater or equal to the number of tasks of the model.
      randomize_expected_outputs: When True, the expected output for each basic
        block and each delta is selected randomly from the list of possible
        values in the basic block proto. Otherwise, takes the first value from
        the list.
      num_prefixes: The number of prefixes in the basic block. If not provided,
        the expected value is added only for the overall prediction.

    Raises:
      ValueError: When `throughputs` does not have the right shape.
    """
    # The blocks should have been cleaned up and there should be one
    # throughput per task. However, that is not enforced anywhere, so we
    # just take self.num_tasks throughputs if there are enough and raise an
    # error if there isn't a sufficient number of throughputs.
    if len(throughputs) < self.num_tasks:
      raise ValueError(
          'Block contains an insufficient number of throughputs:'
          f' {len(throughputs)}, should be {self.num_tasks}'
      )
    block_mask = tuple(
        bool(task_throughput)
        for task_throughput in itertools.islice(throughputs, self.num_tasks)
    )
    block_expected_outputs = []
    if self._use_deltas:
      expected_prefix_throughputs = np.zeros(
          (num_prefixes, self.num_tasks), dtype=self.numpy_dtype
      )
      for task in range(self.num_tasks):
        if throughputs[task] is not None:
          num_prefixes_in_throughputs = len(
              throughputs[task].prefix_inverse_throughput_cycles
          )
          if (
              self._use_delta_loss
              and num_prefixes != num_prefixes_in_throughputs
          ):
            raise ValueError(
                f'Invalid number of prefixes for task {task}.\n'
                f'Expected: {num_prefixes}\n'
                f'Actual:   {num_prefixes_in_throughputs}'
            )
          for prefix_index, prefix_throughputs in enumerate(
              throughputs[task].prefix_inverse_throughput_cycles
          ):
            if randomize_expected_outputs:
              expected_prefix_throughput = random.choice(prefix_throughputs)
            else:
              expected_prefix_throughput = prefix_throughputs[0]
            expected_prefix_throughputs[prefix_index, task] = (
                expected_prefix_throughput
            )
        else:
          expected_prefix_throughputs[:, task] = INVALID_THROUGHPUT_VALUE

      # Compute deltas from prefix throughputs by subtracting the previous
      # element for all elements apart from the first one.
      expected_prefix_throughputs[1:] -= expected_prefix_throughputs[:-1]
      if self._batch_expected_outputs_deltas is None:
        raise ValueError(
            'ModelBase._batch_expected_outputs_deltas is None when using deltas'
            ' to add expected outputs for a basic block in a batch'
        )
      self._batch_expected_outputs_deltas.extend(expected_prefix_throughputs)

    for i in range(self.num_tasks):
      if throughputs[i] is not None:
        cycles = throughputs[i].inverse_throughput_cycles
        # The value of the expected output depends on the mode: when using the
        # whole input, we deterministically use the mean of the expected
        # inverse throughputs of the block. Otherwise, we select a random one.
        if randomize_expected_outputs:
          expected_throughput = random.choice(cycles)
        else:
          expected_throughput = cycles[0]
      else:
        expected_throughput = INVALID_THROUGHPUT_VALUE
      block_expected_outputs.append(expected_throughput)
    if self._batch_expected_outputs is None:
      raise ValueError(
          'ModelBase._batch_expected_outputs is None when adding expected'
          ' outputs for a basic block in a batch'
      )
    self._batch_expected_outputs.append(block_expected_outputs)
    if self._batch_mask is None:
      raise ValueError(
          'ModelBase._batch_mask is None when adding expected outputs for a'
          ' basic block in a batch'
      )
    self._batch_mask.append(block_mask)

  def schedule_batch(
      self,
      basic_blocks: Sequence[BlockOrBlockWithThroughput],
      max_blocks_in_batch: Optional[int] = None,
      max_instructions_in_batch: Optional[int] = None,
      randomize_batch: bool = False,
      randomize_expected_outputs: bool = False,
  ) -> FeedDict:
    """Creates a feed_dict that covers all basic blocks from basic_blocks.

    This method orchestrates everything; it depends on self._start_batch(),
    self._add_basic_block_to_batch(), and self._finalize_batch() for
    model-specific basic block processing.

    The method works in two modes depending on the value of 'randomize_batch':
     - deterministic mode where blocks are added in the order in which they
       appear in basic_blocks, subject to the batch size limits. Blocks that are
       too large to be added to the batch are skipped, and the following blocks
       are considered until max_blocks_in_batch is attained or until all blocks
       from basic_blocks are considered.
     - random mini-batch mode, where the basic blocks are sampled randomly from
       basic_blocks, subject to the batch size limits.

     Similarly, the expected outputs in the batch depend on the value of
     'randomize_expected_outputs'.

     Note that both max_blocks_in_batch and max_instructions_in_batch are upper
     bounds on the size of the batch, but they might not be attained unless the
     sizes of the input basic blocks are perfectly aligned with the limits.

    Args:
      basic_blocks: a list of basic_blocks or basic blocks with throughput. When
        throughputs are provided, the number of entries must correspond to the
        number of tasks learned by the model.
      max_blocks_in_batch: The maximal number of basic blocks in the batch. When
        specified, at most this many basic blocks are added to the batch.
      max_instructions_in_batch: The maximal number of instructions across all
        basic blocks added to the batch.
      randomize_batch: When True, the basic blocks added to the batch are
        selected randomly, without repetition. Otherwise, they are presented in
        the order in which they appear.
      randomize_expected_outputs: When True, the expected output for each basic
        block and each delta is selected randomly from the list of possible
        values in the input structure. Otherwise, takes the first value from the
        list.

    Returns:
      The feed_dict object for the batch.

    Raises:
      ValueError: When `basic_blocks` is empty.
    """
    num_input_blocks = len(basic_blocks)
    if num_input_blocks == 0:
      raise ValueError('basic_blocks must contain at least once block.')

    # The input is a sequence that that contains either only basic blocks with
    # throughputs or only basic blocks without throughputs. We can determine
    # which case it is by looking at the first block of the sequence.
    has_throughputs = isinstance(
        basic_blocks[0], throughput.BasicBlockWithThroughput
    )

    max_blocks_in_batch = max_blocks_in_batch or num_input_blocks
    with timer.scoped('ModelBase.schedule_batch'):
      self._start_batch()

      # Randomize the order/selection of the basic blocks if needed.
      if randomize_batch:
        # When the number of instructions in a batch is limited, we might reject
        # some blocks, because they would increase the size of the batch too
        # much. To mitigate this, we draw more blocks than allowed by the basic
        # block limit, so that we can use them as a replacement if some basic
        # blocks are rejected by the instruction limit.
        # Note that this does not guarantee that the limit will be reached, and
        # it may also skew the distribution towards smaller blocks which are
        # easier to fit into a batch. However, we do not expect this to cause
        # any issues unless the instruction count limit is very tight.
        # TODO(ondrasej): Consider replacing this with a fairer sampling
        # mechanism where we push the rejected blocks to the following batch.
        if max_instructions_in_batch:
          num_blocks_in_sample = min(
              (3 * max_blocks_in_batch) // 2, num_input_blocks
          )
        else:
          num_blocks_in_sample = min(max_blocks_in_batch, num_input_blocks)
        basic_blocks = random.sample(basic_blocks, num_blocks_in_sample)

      num_instructions_in_batch = 0
      num_blocks_in_batch = 0
      for block_or_block_with_throughputs in basic_blocks:
        if num_blocks_in_batch == max_blocks_in_batch:
          break

        block: basic_block.BasicBlock = (
            block_or_block_with_throughputs.block
            if has_throughputs
            else block_or_block_with_throughputs
        )
        if has_throughputs:
          block_with_throughputs: throughput.BasicBlockWithThroughput = (
              block_or_block_with_throughputs
          )

        num_instructions_in_block = len(block.instructions)

        if max_instructions_in_batch:
          if num_instructions_in_block > max_instructions_in_batch:
            # A single basic block has more instruction than what is allowed by
            # the limit on the number of instructions per batch. We skip it and
            # print a warning to the log.
            logging.warn(
                (
                    'A single basic block has more instructions (%d) than'
                    ' max_instructions_in_batch (%d)'
                ),
                num_instructions_in_block,
                max_instructions_in_batch,
            )
            continue
          num_instructions_with_block = (
              num_instructions_in_batch + num_instructions_in_block
          )
          if num_instructions_with_block > max_instructions_in_batch:
            # Adding this basic block to the batch would exceed the maximal
            # number of instructions in the batch.
            continue

        self._add_basic_block_to_batch(block)
        num_prefixes = len(block.instructions)
        if has_throughputs:
          self._add_expected_outputs_to_batch(
              throughputs=block_with_throughputs.throughputs,
              randomize_expected_outputs=randomize_expected_outputs,
              num_prefixes=(num_prefixes if self.use_deltas else None),
          )
        if self._create_delta_block_index:
          assert num_prefixes == num_instructions_in_block
          if self._batch_delta_block_index is None:
            raise ValueError(
                'ModelBase._batch_delta_block_index is None when creating delta'
                ' block index while creating a feed_dict.'
            )
          self._batch_delta_block_index.extend(
              [num_blocks_in_batch] * num_prefixes
          )

        num_instructions_in_batch += num_instructions_in_block
        num_blocks_in_batch += 1

      logging.info(
          'ModelBase.schedule_batch: %d blocks, %d instructions',
          num_blocks_in_batch,
          num_instructions_in_batch,
      )
      return self._finalize_batch(has_throughputs)

  def run_continuous_evaluation(
      self,
      basic_blocks: Sequence[throughput.BasicBlockWithThroughput],
      checkpoint_dir: str,
      summary_dir: str,
      tf_master: str = '',
      eval_interval_seconds: int = 60,
      max_num_evaluations: Optional[int] = None,
      session_hooks: Optional[
          Sequence[tf.compat.v1.train.SessionRunHook]
      ] = None,
      max_blocks_in_batch: Optional[int] = None,
      max_instructions_in_batch: Optional[int] = None,
  ) -> None:
    """Runs continuous evaluation of the model on the given basic blocks.

    Reloads the most recent version of the trained model from a given directory,
    computes the loss on the provided data set and updates summaries of global
    step and loss in a provided summary directory.

    Args:
      basic_blocks: A collection of basic blocks that are used for the
        evaluation. All blocks in the collection are used in each step of the
        evaluation.
      checkpoint_dir: The checkpoint directory for the model. Trained models are
        loaded from this directory.
      summary_dir: The summary directory for the model. The summaries are
        written to the files in this directory.
      tf_master: The address of the TensorFlow master.
      eval_interval_seconds: The number of seconds after which the evaluation is
        repeated.
      max_num_evaluations: The maximal number of evaluations to run. When None,
        the evaluation is ran indefinitely.
      session_hooks: An optional list of session hooks that are used in the
        evaluation session.
      max_blocks_in_batch: The maximal number of basic blocks used in the
        evaluation batch.
      max_instructions_in_batch: The maximal number of instructions used in the
        evaluation batch.
    """
    schedule = self.schedule_batch(
        basic_blocks,
        max_blocks_in_batch=max_blocks_in_batch,
        max_instructions_in_batch=max_instructions_in_batch,
        randomize_batch=False,
        randomize_expected_outputs=False,
    )

    hooks = [
        tf_slim.evaluation.StopAfterNEvalsHook(1),
        tf_slim.evaluation.SummaryAtEndHook(summary_dir, feed_dict=schedule),
        # Save the models with the best MAPE.
        # We disable attribute error detection on self._loss because it is
        # nullable and pytype expects there to be a check here.
        SaveBestCheckpoint(
            error_tensor=self._loss.mean_absolute_percentage_error,  # pytype: disable=attribute-error
            checkpoint_dir=os.path.join(summary_dir, 'best_models'),
            global_step=self.global_step,
        ),
    ]

    if session_hooks:
      hooks.extend(session_hooks)
    logging.info('Starting continuous evaluation.')
    tf_slim.evaluation.evaluate_repeatedly(
        checkpoint_dir,
        master=tf_master,
        eval_ops=[
            self.global_step,
            self._loss_tensor,
        ],
        feed_dict=schedule,
        hooks=hooks,
        max_number_of_evaluations=max_num_evaluations,
        eval_interval_secs=eval_interval_seconds,
    )

  def predict(
      self,
      basic_blocks: Iterable[basic_block.BasicBlock],
      max_blocks_in_batch: Optional[int] = None,
      max_instructions_in_batch: Optional[int] = None,
  ) -> Iterable[throughput.BasicBlockWithThroughput]:
    """Predicts the inverse throughput using the model.

    The input sequence is iterated through only once, and the method may be
    used with basic block sources such as tf.io.tf_record_iterator.

    Args:
      basic_blocks: The collection of basic blocks for which the inverse
        throughput is predicted.
      max_blocks_in_batch: The maximal number of basic blocks processed in a
        single batch. When not specified, the number of basic blocks in a batch
        is unlimited.
      max_instructions_in_batch: The maximal number of instructions across all
        basic blocks processed in a single batch. When not specified, the number
        of instructions in a batch is unlimited.

    Yields:
      The basic blocks from basic_blocks. Each basic block has a new
      inverse_throughputs value added to it with the prediction from the model.
    """
    batches = training.batches(
        basic_blocks,
        get_num_instructions=training.get_num_instructions_in_block,
        max_blocks_in_batch=max_blocks_in_batch,
        max_instructions_in_batch=max_instructions_in_batch,
    )
    for batch_index, batch in enumerate(batches):
      logging.info('Processing batch %d (%d blocks).', batch_index, len(batch))
      batch_output_blocks = []
      with timer.scoped('ModelBase.predict - one batch'):
        schedule = self.schedule_batch(batch)
        output_dict = self(schedule)
        output = output_dict['output']
        if self._use_deltas:
          output_deltas = output_dict['output_deltas']
          output_index = 0
          for block_index, block in enumerate(batch):
            block_len = len(block.instructions)
            # Extract the per-instruction throughput predictions for the basic
            # block. This has shape (block_len, num_tasks).
            block_output_deltas = output_deltas[
                output_index : output_index + block_len
            ]
            assert block_output_deltas.shape == (block_len, self.num_tasks)
            throughputs = []
            for task_index in range(self.num_tasks):
              task_output_deltas = block_output_deltas[:, task_index]
              task_throughputs = throughput.BasicBlockThroughput(
                  inverse_throughput_cycles=(output[block_index, task_index],),
                  prefix_inverse_throughput_cycles=tuple(
                      (delta_throughput,)
                      for delta_throughput in task_output_deltas
                  ),
              )
              throughputs.append(task_throughputs)
            output_index += block_len
            batch_output_blocks.append(
                throughput.BasicBlockWithThroughput(
                    block=block, throughputs=throughputs
                )
            )
        else:
          for block_index, block in enumerate(batch):
            throughputs = []
            for task_index in range(self.num_tasks):
              throughputs.append(
                  throughput.BasicBlockThroughput(
                      inverse_throughput_cycles=(
                          output[block_index, task_index],
                      )
                  )
              )
            batch_output_blocks.append(
                throughput.BasicBlockWithThroughput(
                    block=block, throughputs=throughputs
                )
            )

      for output_block in batch_output_blocks:
        yield output_block

  def train(
      self,
      basic_block_list: Sequence[throughput.BasicBlockWithThroughput],
      num_epochs: int,
      max_blocks_in_batch: Optional[int],
      max_instructions_in_batch: Optional[int],
      randomize_batches: bool = True,
      randomize_expected_outputs: bool = False,
      hooks: Sequence[tuple[int, Callable[[], None]]] | None = None,
  ) -> Optional[training.TrainingEpochStats]:
    """Runs training of the model on the given training data.

    Args:
      basic_block_list: The collection of input basic blocks.
      num_epochs: The number of training steps. This value is used only for
        profiling and logging; the method uses monitored_session.should_stop()
        to decide when to stop the training.
      max_blocks_in_batch: The maximal number of basic blocks in a single batch;
        when not specified, the maximal number of basic blocks in the batch is
        unlimited.
      max_instructions_in_batch: The maximal number of instructions in a single
        batch; when not specified, the number of instructions in a batch is
        unlimited.
      randomize_batches: Set to True to randomize the basic blocks in the
        batches used for training. When False, the model is trained with batches
        that cycle through the blocks in basic_block_list in the order in which
        they appear in the list.
      randomize_expected_outputs: Set to True to randomly select the expected
        outputs used for training from the available values. When False, it
        takes the first value from the list.
      hooks: Hooks to run during the training process.

    Returns:
      The loss before the last training step. Returns None when no training was
      performed.
    """
    if randomize_batches:

      def run_one_epoch():
        return self.train_mini_batch(
            basic_block_list,
            max_blocks_in_batch=max_blocks_in_batch,
            max_instructions_in_batch=max_instructions_in_batch,
            randomize_expected_outputs=randomize_expected_outputs,
        )

    else:
      # Creates an infinite list of batches that respect the limits and that
      # cycle through the basic blocks in the input list. When the limit on the
      # number of basic blocks per batch is not specified, we set it so that in
      # each step we train on basic_block_list, with no repetitions.
      max_blocks_in_batch = max_blocks_in_batch or len(basic_block_list)
      batches = iter(
          training.batches(
              itertools.cycle(basic_block_list),
              get_num_instructions=(
                  training.get_num_instructions_in_block_with_throughput
              ),
              max_blocks_in_batch=max_blocks_in_batch,
              max_instructions_in_batch=max_instructions_in_batch,
          )
      )

      def run_one_epoch():
        batch = next(batches)
        schedule = self.schedule_batch(
            batch, randomize_expected_outputs=randomize_expected_outputs
        )
        return self.train_batch(schedule)

    with timer.scoped('ModelBase.train - one batch', num_iterations=num_epochs):
      for epoch_index in range(num_epochs):
        tf.summary.experimental.set_step(epoch_index)
        stats = run_one_epoch()
        logging.info('Training: %s', stats)
        if not hooks:
          continue
        for epochs_every, hook_function in hooks:
          if (epoch_index + 1) % epochs_every == 0:
            hook_function()
      return stats

  def _compute_loss(self, schedule: FeedDict) -> loss_utils.LossComputation:
    output = self(schedule, train=True)
    loss = loss_utils.LossComputation(
        output['output'],
        tf.constant(schedule['expected_outputs']),
        tf.constant(schedule['output_mask']),
        percentile_ranks=self._collected_percentile_ranks,
        dtype=self.dtype,
    )

    if self._use_deltas:
      self._delta_loss = loss_utils.LossComputation(
          output['output_deltas'],
          tf.constant(schedule['expected_outputs_deltas']),
          output['output_mask_deltas'],
          percentile_ranks=self._collected_percentile_ranks,
          dtype=self.dtype,
      )

      if self._use_delta_loss:
        loss = self._delta_loss

    return loss

  def compute_loss_tensor(self, schedule: FeedDict):
    return tf.reduce_mean(
        self._compute_loss(schedule).loss_tensor(
            self._loss_normalization, self._loss_type
        )
    )

  def _get_trainable_variables(self):
    return self.trainable_variables

  def train_batch(
      self,
      schedule: FeedDict,
  ) -> training.TrainingEpochStats:
    """Trains a batch based on the given schedule.

    Args:
      schedule: A feed_dict that describes the current batch.

    Returns:
      The loss on the training set before the training step.
    """
    with timer.scoped('ModelBase.train_batch'):
      # The keys of stats are the names of keyword arguments of the constructor
      # of TraningEpochStats. This dict can then be unpacked to
      # TrainingEpochStats.__init__() as keyword arguments.
      with tf.GradientTape() as tape:
        stats = {}
        loss = self._compute_loss(schedule)
        loss_tensor_per_task = loss.loss_tensor(
            self._loss_normalization, self._loss_type
        )
        loss_tensor = tf.reduce_mean(loss_tensor_per_task)

        # The list of variables to optimize. By default, the list is empty which
        # means optimize all trainable variables.
        requested_variables = set()
        for variable_group in self._trained_variable_groups:
          requested_variables.update(
              variable.ref()
              for variable in self._variable_groups.get(variable_group)
          )

        trainable_variables = self._get_trainable_variables()
        variables = (
            [variable.deref() for variable in requested_variables]
            if requested_variables
            else trainable_variables
        )

        grads = tape.gradient(loss_tensor, variables)
        grads_and_vars = zip(grads, variables)

      # TODO(vbshah): Compute and log the number of steps per second as well.
      tf.summary.scalar('learning_rate', self._decayed_learning_rate)
      tf.summary.scalar('overall_loss', loss_tensor)

      # TODO(vbshah): Consider writing delta loss summaries as well.
      self._add_error_summaries('absolute_mse', loss.mean_squared_error)
      self._add_error_summaries(
          'relative_mae',
          loss.mean_absolute_percentage_error,
      )
      self._add_error_summaries(
          'relative_mse',
          loss.mean_squared_percentage_error,
      )
      self._add_percentile_summaries(
          'absolute_error',
          self._collected_percentile_ranks,
          loss.absolute_error_percentiles,
      )
      self._add_percentile_summaries(
          'absolute_percentage_error',
          self._collected_percentile_ranks,
          loss.absolute_percentage_error_percentiles,
      )

      stats['loss'] = loss_tensor
      stats['epoch'] = self.global_step
      stats['absolute_mse'] = loss.mean_squared_error
      stats['relative_mae'] = loss.mean_absolute_percentage_error
      stats['relative_mse'] = loss.mean_squared_percentage_error
      stats['absolute_error_percentiles'] = loss.absolute_error_percentiles
      stats['relative_error_percentiles'] = (
          loss.absolute_percentage_error_percentiles
      )

      if self._grad_clip_norm:
        if self._grad_clip_norm <= 0.0:
          logging.warning(
              'The gradients are clipped to zero. Please revise if this is not '
              'intended.'
          )
        grads_and_vars = [
            (self._clip_if_not_none(g), v) for g, v in grads_and_vars
        ]
      self._train_step = self._optimizer.apply_gradients(
          grads_and_vars, global_step=self.global_step
      )

      return training.TrainingEpochStats(
          percentile_ranks=self._collected_percentile_ranks, **stats
      )

  def train_mini_batch(
      self,
      basic_blocks: Sequence[throughput.BasicBlockWithThroughput],
      max_blocks_in_batch: int,
      max_instructions_in_batch: Optional[int] = None,
      randomize_expected_outputs: bool = False,
  ) -> training.TrainingEpochStats:
    """Trains a random mini-batch selected from basic_blocks.

    Args:
      sess: The TensorFlow session the training is running in.
      basic_blocks: A collection of basic blocks from which the mini-batch is
        taken.
      max_blocks_in_batch: The maximal number of basic blocks in a single batch.
        When not specified, the number of blocks is not limited.
      max_instructions_in_batch: The maximal number of instructions in a single
        batch. When not specified, the number of instructions in a batch is not
        limited.
      randomize_expected_outputs: Set to True to randomly select the expected
        values used for training from the available values. When False, the
        expected value is computed as the mean of the available values.

    Returns:
      The loss on the mini batch before the training step.
    """
    train_schedule = self.schedule_batch(
        basic_blocks,
        max_blocks_in_batch=max_blocks_in_batch,
        max_instructions_in_batch=max_instructions_in_batch,
        randomize_batch=True,
        randomize_expected_outputs=randomize_expected_outputs,
    )
    return self.train_batch(train_schedule)
