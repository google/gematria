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
"""A generic main function and command-line flags for Gematria models.

This library provides a generic function for running Gematria machine learning
models + definitions of command-line flags that are common to all models.

Typical usage:
  def main(_):
    main_function.run_gematria_model_from_command_line_flags(
        sequence_model_hlstm.HierarchicalLstmModel,
        ...
    )
"""

from collections.abc import Iterable, Mapping, Sequence
import functools
import os
import random
import re
import sys
from typing import Any, Type

from absl import flags
from absl import logging
from gematria.basic_block.python import throughput
from gematria.basic_block.python import throughput_protos
from gematria.io.python import gfile_copy
from gematria.io.python import options as io_options
from gematria.io.python import tfrecord
from gematria.io.python import utils
from gematria.model.python import inference
from gematria.model.python import model_base
from gematria.model.python import options as model_options
from gematria.model.python import training
from gematria.proto import throughput_pb2
from gematria.utils.python import timer
import numpy as np
import tensorflow.compat.v1 as tf

_ACTION = flags.DEFINE_enum_class(
    'gematria_action',
    model_options.Action.TRAIN,
    model_options.Action,
    'The action to be performed by the Gematria main function.',
)

_INPUT_FILES = flags.DEFINE_list(
    'gematria_input_file',
    [],
    'The TFRecord files from which basic block data is loaded.',
)
_INPUT_FILE_SCALING = flags.DEFINE_float(
    'gematria_input_file_scaling',
    1.0,
    (
        'Scaling factor applied to the inverse throughputs in data loaded from'
        ' --gematria_input_file.'
    ),
)

_GRAD_CLIP_NORM = flags.DEFINE_float(
    'gematria_grad_clip_norm',
    None,
    (
        'The maximum clipping value to clip gradients during training. If None,'
        ' no clipping will be applied.'
    ),
)

_LOAD_GLOBAL_STEP_FROM_CKPT = flags.DEFINE_bool(
    'gematria_load_global_step_from_ckpt',
    False,
    'If True, the global step value is also loaded from the given checkpoint.',
)

_THROUGHPUT_SOURCE_FILTERS = flags.DEFINE_multi_string(
    'gematria_throughput_source_filter',
    ['.*'],
    (
        'A list of regular expressions used to filter inverse throughput'
        ' sources which are preserved in the input file. When empty, all'
        ' inverse throughputs are preserved.'
    ),
)
_TASK_NAMES = flags.DEFINE_multi_string(
    'gematria_task_names',
    [],
    (
        'The names of the tasks used in the multi-task training environment.'
        ' When non-empty, this list must be parallel with'
        ' --gematria_throughput_source_filter, i.e. both lists must have the'
        ' same number of elements, and the elements at the same index'
        ' correspond to the same task.'
    ),
)
_MODEL_NAME = flags.DEFINE_string(
    'gematria_model_name',
    None,
    (
        'The name of the model used in the output BasicBlockWithThroughputProto'
        ' when running with --gematria_action=predict.'
    ),
)
_GEMATRIA_OUTPUT_FILE = flags.DEFINE_string(
    'gematria_output_file',
    '',
    (
        'The TFRecord file to which the processed basic blocks are stored. This'
        ' option is used only when processing basic blocks with a trained'
        ' model, i.e. the --gematria_action flag is "predict".'
    ),
)

# NOTE(ondrasej): The name 'master' is required by the distributed training
# templates provided by the TensorFlow team.
_MASTER = flags.DEFINE_string(
    'master',
    '',
    'The BNS of the TensorFlow runtime to use for the processing.',
)
_GEMATRIA_MAX_BLOCKS_IN_BATCH = flags.DEFINE_integer(
    'gematria_max_blocks_in_batch',
    None,
    (
        'The maximal number of basic blocks in a single batch. In training'
        ' mode, this option also triggers mini-batch training, and the size of'
        ' each batch does not exceed this limit. In eval mode, the evaluation'
        ' will be done on a single batch whose size does not exceed this limit.'
        ' In prediction mode the input is split into batches that preserve the'
        ' original order of the basic blocks and where none of the batches'
        ' exceeds this limit'
    ),
)
_GEMATRIA_MAX_INSTRUCTIONS_IN_BATCH = flags.DEFINE_integer(
    'gematria_max_instructions_in_batch',
    1000000,
    (
        'The maximal number of instructions processed in a single batch across'
        ' all basic blocks in the batch. Providing the limit avoids crashes due'
        ' to running out of memory or by running into the maximal size of a'
        ' serialized protocol buffers when processing data sets with large'
        ' basic blocks. In training mode, this puts an additional constraint to'
        ' batch selection, and may raduce the number of basic blocks in a given'
        ' batch below --gematria_training_max_blocks_in_batch. In eval mode,'
        ' this option may cause some basic blocks to be dropped when they would'
        ' exceed the instruction limit. In prediction mode, the batches may be'
        ' dowsized to fit into the limit; basic blocks with more instructions'
        ' than the value of this flag will be dropped.'
    ),
)
_GEMATRIA_TRAINING_NUM_EPOCHS = flags.DEFINE_integer(
    'gematria_training_num_epochs',
    10000,
    'The number of training epochs to run.',
)
_GEMATRIA_TRAINING_RANDOMIZE_BATCHES = flags.DEFINE_bool(
    'gematria_training_randomize_batches',
    True,
    (
        'Use randomized batches when training the model. When set to True, each'
        ' batch is taken as a random sample from the training data set;'
        ' otherwise, the training algorithm cycles through all basic blocks in'
        ' the order in which they appear in the input.'
    ),
)
_DROP_INVALID_BLOCKS = flags.DEFINE_bool(
    'gematria_drop_invalid_blocks',
    False,
    (
        'Drop basic blocks that fail model-specific validation. By default,'
        ' this is turned off to avoid errors that are silently forgotten. It'
        ' can be overridden to allow processing external data sets in inference'
        ' mode.'
    ),
)
_TRAINING_THROUGHPUT_SELECTION = flags.DEFINE_enum_class(
    'gematria_training_throughput_selection',
    io_options.ThroughputSelection.MEAN,
    io_options.ThroughputSelection,
    (
        'The way throughput values are selected during training and evaluation'
        ' when a basic block has more than one inverse throughput value.'
    ),
)
_GEMATRIA_NUMPY_PRINT_EDGEITEMS = flags.DEFINE_integer(
    'gematria_numpy_print_edgeitems',
    20,
    'The number of learned values to display for the list.',
)
_GEMATRIA_LOG_DEVICE_PLACEMENT = flags.DEFINE_bool(
    'gematria_log_device_placement',
    False,
    'Print TensorFlow op placement to devices to the log.',
)
_GEMATRIA_RANDOM_SEED = flags.DEFINE_integer(
    'gematria_random_seed',
    123456789,
    (
        'The seed used to initialize the random number generators. When'
        ' negative, the default (possibly random) seed is used.'
    ),
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    'gematria_checkpoint_dir',
    '',
    'The directory in which checkpoints are stored durinng the training.',
)
_GEMATRIA_CHECKPOINT_MAX_TO_KEEP = flags.DEFINE_integer(
    'gematria_checkpoint_max_to_keep',
    100,
    'The maximum number of latest checkpoints to store during the training.',
)
_CHECKPOINT_FILE = flags.DEFINE_string(
    'gematria_checkpoint_file',
    '',
    (
        'A checkpoint file from which a trained model is restored. Used when '
        '--gematria_action is "predict".'
    ),
)

_WARMSTART_FILE = flags.DEFINE_string(
    'gematria_warmstart_checkpoint_file',
    None,
    (
        'A checkpoint file from which training of a model can be warm-started.'
        ' When --gematria_action is "train" and there is no checkpoint in the'
        ' directory specified through --gematria_checkpoint_dir, this'
        ' checkpoint is partially restored via'
        ' training.partially_restore_from_checkpoint. This is mutually'
        ' exclusive with --gematria_warmstart_checkpoint_dir.'
    ),
)
_WARMSTART_DIR = flags.DEFINE_string(
    'gematria_warmstart_checkpoint_dir',
    None,
    (
        'A checkpoint directory from which training of a model can be'
        ' warm-started. When --gematria_action is "train" and there is no'
        ' checkpoint in the directory specified through'
        ' --gematria_checkpoint_dir, the lateste checkpoint from this directory'
        ' is partially restored via training.partially_restore_from_checkpoint.'
        ' This is mutually exclusive with --gematria_warmstart_checkpoint_file.'
    ),
)

_RESUME_FROM_DIR = flags.DEFINE_string(
    'gematria_resume_from_dir',
    None,
    (
        'An experiment directory to resume from. This flag provides a way to'
        ' resume computation from an existing checkpoint directory without'
        ' writing to that directory. When no checkpoint exists in the'
        ' checkpoint directory of the current experiment, all contents is'
        ' copied from --gematria_resume_from_dir to --gematria_resume_to_dir.'
        ' This is stronger than using a warmstart checkpoint: 1. it restores'
        ' the contents of all variables in the model, not just trainable ones;'
        ' 2. it preserves all events and checkpoint files from the source'
        ' directory; 3. it requires that the TensorFlow graphs on both ends are'
        ' the same. Used only when --gematria_action is "train".'
    ),
)
_RESUME_TO_DIR = flags.DEFINE_string(
    'gematria_resume_to_dir',
    None,
    (
        'The directory to copy the contents of --gematria_resume_from_dir to.'
        ' When specified, --gematria_resume_to_dir must be a parent directory'
        ' of --gematria_checkpoint_dir or they must be the same directory.'
    ),
)

_COLLECTED_PERCENTILE_RANKS = flags.DEFINE_list(
    'gematria_collected_percentile_ranks',
    ['50', '90', '95', '99'],
    (
        'The ranks of the percentiles of the absolute and relative errors '
        'collected by the model.'
    ),
)
_GEMATRIA_SUMMARY_DIR = flags.DEFINE_string(
    'gematria_summary_dir',
    '',
    'The directory to which the summaries from the training are stored.',
)
_GEMATRIA_SAVE_CHECKPOINT_SECS = flags.DEFINE_integer(
    'gematria_save_checkpoint_secs',
    60,
    'The number of seconds of training after which a checkpoint is saved.',
)
_GEMATRIA_SAVE_SUMMARIES_SECS = flags.DEFINE_integer(
    'gematria_save_summaries_secs',
    60,
    'The number of seconds of training after which summaries are saved.',
)
_GEMATRIA_EVAL_INTERVAL_SECS = flags.DEFINE_integer(
    'gematria_eval_interval_secs',
    60,
    (
        'When running in the continuous evaluation mode, this is the number of '
        'seconds between two consecutive evaluations.'
    ),
)
_GEMATRIA_TRAINING_TASK = flags.DEFINE_integer(
    'gematria_training_task',
    0,
    'The ID of the training task when running on Borg.',
)
_GEMATRIA_TRAINING_PS_TASKS = flags.DEFINE_integer(
    'gematria_training_ps_tasks',
    0,
    (
        'The number of parameter server tasks used when running in a'
        ' distributed training setup.'
    ),
)
_GEMATRIA_SYNCHRONOUS_TRAINING = flags.DEFINE_bool(
    'gematria_synchronous_training',
    True,
    (
        'Use synchronous weight updates when running in a distributed training '
        'setup.'
    ),
)
_LOSS_NORMALIZATION = flags.DEFINE_enum_class(
    'gematria_loss_normalization',
    model_options.ErrorNormalization.NONE,
    model_options.ErrorNormalization,
    (
        'The type of normalization applied to the errors when computing the'
        ' loss. The value must be one of the values of the ErrorNormalization'
        ' enum.'
    ),
)
_LOSS_TYPE = flags.DEFINE_enum_class(
    'gematria_loss_type',
    model_options.LossType.MEAN_SQUARED_ERROR,
    model_options.LossType,
    (
        'The type of loss used to train the model. The value must be one of the'
        ' values of the LossType enum.'
    ),
)
_TRAINED_VARIABLES = flags.DEFINE_list(
    'gematria_trained_variables',
    None,
    (
        'The list of variable groups in the model that are updated during the'
        ' training. This allows restricting training to only a part of the'
        ' model. The names of the variable groups are model-specific; when'
        ' unspecified, all variables in the model are trained.'
    ),
)
_LEARNING_RATE = flags.DEFINE_float(
    'gematria_learning_rate',
    0.001,
    (
        'The learning rate of the optimizer used when training the model. When'
        ' not specified, the default learning rate of the optimizer is used.'
    ),
)
_OPTIMIZER_TYPE = flags.DEFINE_enum_class(
    'gematria_optimizer_type',
    enum_class=model_options.OptimizerType,
    default=model_options.OptimizerType.ADAM,
    help='Specifies the optimizer type for training. Default Adam optimizer.',
)
_DECAY_STEPS = flags.DEFINE_integer(
    'gematria_decay_steps',
    0,
    (
        'Specifies the maximum number of decay steps. The effect of decay steps'
        ' depends on the learning rate schedule. If a learning rate schedule is'
        ' selected and `gematria_decay_steps` is zero, a `ValueError` is'
        ' raised.'
    ),
)
_DECAY_RATE = flags.DEFINE_float(
    'gematria_decay_rate',
    0.0,
    (
        'Specifies the rate at which the learning rate is decayed. The effect'
        ' of decay rate depends on the learning rate schedule.'
    ),
)
_LEARNING_RATE_SCHEDULE = flags.DEFINE_enum_class(
    'gematria_learning_rate_schedule',
    enum_class=model_options.LearningRateScheduleType,
    default=model_options.LearningRateScheduleType.NONE,
    help='Specifies the learning type schedule for decaying the learning rate.',
)
_GEMATRIA_NUM_TRAINING_WORKER_REPLICAS = flags.DEFINE_integer(
    'gematria_num_training_worker_replicas',
    1,
    'The number of training worker replicas in the distributed training setup.',
)
_GEMATRIA_NUM_TRAINING_WORKER_REPLICAS_TO_AGGREGATE = flags.DEFINE_integer(
    'gematria_num_training_worker_replicas_to_aggregate',
    1,
    (
        'When running distributed training with synchronous weight updats, this'
        ' is the (minimal) number of replicas that is aggregated into a single'
        ' step of the training. Keeping this number lower than the total number'
        " of replicas makes the training faster as it doesn't have to wait for"
        ' the slowest replica.'
    ),
)
_GRAPH_DEF_FILE = flags.DEFINE_string(
    'gematria_graph_def_file',
    None,
    (
        'When running in export graph def node, this is the name of the file to'
        ' which the GraphDef proto for the model is written. The proto is'
        ' stored in the text format.'
    ),
)
_GEMATRIA_USE_SEQ2SEQ_LOSS = flags.DEFINE_bool(
    'gematria_use_seq2seq_loss',
    True,
    (
        'When running in seq2seq mode, determines how the loss is computed:'
        ' when True, the loss is computed from prediction errors on each prefix'
        ' of each basic block; when False, the loss is computed from the'
        ' overall error of the basic block. This flag has no effect for models'
        ' that are not seq2seq.'
    ),
)


@flags.validator(
    _COLLECTED_PERCENTILE_RANKS.name,
    '--gematria_collected_percentile_ranks must be a list of integers',
)
def _is_list_of_ints(values: Sequence[str]) -> bool:
  """Checks that all values in a sequence are integers."""
  try:
    map(int, values)
    return True
  except ValueError:
    return False


@flags.multi_flags_validator(
    (_ACTION.name, _TASK_NAMES.name, _THROUGHPUT_SOURCE_FILTERS.name),
    (
        '--gematria_task_names must either be empty or it must have the same'
        ' number of elements as --gematria_throughput_source_filter'
    ),
)
def _task_names_match_throughput_source_filters(
    flags_dict: Mapping[str, Sequence[str]]
) -> bool:
  """Checks that the numbers of tasks and throughput source filtes match."""
  if flags_dict[_ACTION.name] in model_options.ACTIONS_WITHOUT_INPUT_DATA:
    # In these modes, we do not match throughputs from the input data with
    # outputs of the model, and so we do not need the throughput source filters
    # to match with tasks.
    return True
  task_names = flags_dict[_TASK_NAMES.name]
  filters = flags_dict[_THROUGHPUT_SOURCE_FILTERS.name]
  return not task_names or len(task_names) == len(filters)


@flags.multi_flags_validator(
    (_WARMSTART_DIR.name, _WARMSTART_FILE.name),
    (
        f'At most one of {_WARMSTART_DIR.name} and {_WARMSTART_FILE.name} can'
        ' be used when training a model'
    ),
)
def _at_most_one_warm_start_source_can_be_used(
    flags_dict: Mapping[str, str]
) -> bool:
  """Checks that at most one warm start flag is used."""
  warmstart_dir = flags_dict[_WARMSTART_DIR.name]
  warmstart_file = flags_dict[_WARMSTART_FILE.name]
  return not warmstart_dir or not warmstart_file


@flags.multi_flags_validator(
    (_RESUME_FROM_DIR.name, _RESUME_TO_DIR.name),
    (
        f'{_RESUME_FROM_DIR.name} and {_RESUME_TO_DIR.name} must be either both'
        ' used or both unused.'
    ),
)
def _resume_from_and_resume_to_dir_must_be_used_at_the_same_time(
    flags_dict: Mapping[str, str]
) -> bool:
  """Checks that the resume dir flags are either both present or both absent."""
  resume_from_dir = flags_dict[_RESUME_FROM_DIR.name]
  resume_to_dir = flags_dict[_RESUME_TO_DIR.name]
  return bool(resume_from_dir) == bool(resume_to_dir)


def _warmstart_from_file(scaffold: tf.train.Scaffold, sess):
  """Warmstarts the model from a specific checkpoint."""
  del scaffold  # Unused.
  if not tf.io.gfile.exists(f'{_WARMSTART_FILE.value}.index'):
    raise ValueError(f'No checkpoint was found at "{_WARMSTART_FILE.value}"')
  training.partially_restore_from_checkpoint(
      _WARMSTART_FILE.value, _LOAD_GLOBAL_STEP_FROM_CKPT.value, sess
  )


def _warmstart_from_dir(scaffold: tf.train.Scaffold, sess):
  """Warmstarts the model from the latest checkpoint in a directory."""
  del scaffold  # Unused.
  checkpoint = tf.train.latest_checkpoint(_WARMSTART_DIR.value)
  if not checkpoint:
    raise ValueError(f'No checkpoint was found at "{_WARMSTART_DIR.value}"')
  training.partially_restore_from_checkpoint(
      checkpoint, _LOAD_GLOBAL_STEP_FROM_CKPT.value, sess
  )


def _monitored_training_session_from_flags(
    model: model_base.ModelBase, is_chief: bool
) -> tf.train.MonitoredTrainingSession:
  """Creates a monitored training session for 'model' from command-line flags.

  Args:
    model: The model for which the session is created.
    is_chief: True when this is the chief training worker in a distributed
      setup.

  Returns:
    The monitored training session object.
  """
  hooks = []
  if _GEMATRIA_TRAINING_NUM_EPOCHS.value > 0:
    hooks.append(
        tf.train.StopAtStepHook(last_step=_GEMATRIA_TRAINING_NUM_EPOCHS.value)
    )
  hooks += model.get_monitored_training_session_hooks()
  session_config = tf.ConfigProto(
      log_device_placement=_GEMATRIA_LOG_DEVICE_PLACEMENT.value
  )
  scaffold_init_fn = None
  if _WARMSTART_FILE.value:
    # If there is a checkpoint to bootstrap from, we add an init_fn to the
    # monitored session that restores it. This init_fn is called only when an
    # actual checkpoint is not available to fully restore the model.
    scaffold_init_fn = _warmstart_from_file
  elif _WARMSTART_DIR.value:
    # If there is a directory to bootstrap from, we find the latest checkpoint
    # in this directory and add an init_fn to the monitored session the same way
    # as with _WARMSTART_FILE above.
    scaffold_init_fn = _warmstart_from_dir

  scaffold = tf.train.Scaffold(
      init_fn=scaffold_init_fn,
      saver=tf.train.Saver(
          max_to_keep=_GEMATRIA_CHECKPOINT_MAX_TO_KEEP.value,
          keep_checkpoint_every_n_hours=1,
      ),
  )

  return tf.train.MonitoredTrainingSession(
      checkpoint_dir=_CHECKPOINT_DIR.value,
      config=session_config,
      scaffold=scaffold,
      hooks=hooks,
      is_chief=is_chief,
      master=_MASTER.value,
      save_checkpoint_secs=_GEMATRIA_SAVE_CHECKPOINT_SECS.value,
      save_summaries_secs=_GEMATRIA_SAVE_SUMMARIES_SECS.value,
      summary_dir=_GEMATRIA_SUMMARY_DIR.value,
  )


def _resume_previous_experiment_if_needed():
  """Carries over data from a previous experiment if requested by flags.

  When resume directories are specified through --gematria_resume_from_dir and
  --gematria_resume_to_dir, and the checkpoint directory specified via
  --gematria_checkpoint_dir is empty, copies all contents of the directory
  specified in --gematria_resume_from_dir to the path specified in
  --gematria_resume_to_dir. Expects that --gematria_checkpoint_dir is a
  subdirectory of --gematria_resume_to_dir or that they are the same directory.

  When copying, inspects all files called `checkpoint` in the copied directories
  and replaces the path of the old directory with the path of the new Directory
  using a string replace.

  Does nothing when:
    * --gematria_resume_from_dir is not specified,
    * The directory specified via --gematria_checkpoint_dir is not empty.

  Raises:
    ValueError: When --gematria_resume_from_dir is provided without also
      providing --gematria_resume_to_dir, or when --gematria_checkpoint_dir lies
      outside of --gematria_resume_to_dir.
  """
  if not _RESUME_FROM_DIR.value:
    return
  if not _RESUME_TO_DIR.value:
    raise ValueError(
        '--gematria_resume_from_dir was provided without providing'
        ' also --gematria_resume_to_dir'
    )
  checkpoint_dir_contents = tf.io.gfile.glob(
      os.path.join(_CHECKPOINT_DIR.value, '*')
  )
  if checkpoint_dir_contents:
    # Our own checkpoint dir is non-empty.
    logging.info('Checkpoint dir is non-empty, no need to resume.')
    return

  common_path = os.path.commonpath(
      (_CHECKPOINT_DIR.value, _RESUME_TO_DIR.value)
  )
  if os.path.abspath(common_path) != os.path.abspath(_RESUME_TO_DIR.value):
    raise ValueError(
        f'--gematria_checkpoint_dir ({_CHECKPOINT_DIR.value}) is not a'
        ' subdirectory of --gematria_resume_to_dir ({_RESUME_TO_DIR.value})'
    )

  logging.info(
      'Resuming training from %s to %s.',
      _RESUME_FROM_DIR.value,
      _RESUME_TO_DIR.value,
  )
  gfile_copy.copy_dir(
      _RESUME_FROM_DIR.value, _RESUME_TO_DIR.value, overwrite=True
  )
  logging.info('Resuming has finished, hacking checkpoint file(s).')
  # The `checkpoint` file contains absolute paths to individual checkpoints. We
  # need to replace the old paths with the new ones so that the new experiment
  # does not touch (and eventually delete) files of the old experiment.
  for dirname, _, file_names in tf.io.gfile.walk(_RESUME_TO_DIR.value):
    for filename in file_names:
      if filename != 'checkpoint':
        continue
      checkpoint_filename = os.path.join(dirname, filename)
      with tf.io.gfile.GFile(checkpoint_filename, 'r') as checkpoint_file:
        contents = checkpoint_file.read()
      normalized_from_dir = os.path.abspath(_RESUME_FROM_DIR.value)
      normalized_to_dir = os.path.abspath(_RESUME_TO_DIR.value)
      contents = contents.replace(normalized_from_dir, normalized_to_dir)
      with tf.io.gfile.GFile(checkpoint_filename, 'w') as checkpoint_file:
        checkpoint_file.write(contents)
  logging.info('Hacking checkpoint files done.')


def _make_basic_block_reader_from_command_line_flags(
    input_files: Sequence[str], source_filter_list: Sequence[str]
) -> Iterable[throughput_pb2.BasicBlockWithThroughputProto]:
  """Creates a reader that enumerates basic block protos from the input file.

  Reads the basic blocks from a collection of TFRecord files file `input_files`.
  Apart from reading the protos, the reader does the following transformations:
    * when source_filter is not empty, it modifies the inverse throughput data
      for each block so that the number of throughputs in the proto is the same
      as `len(source_filter_list)` and the throughput source at index i matches
      filter at the same index. When there is no throughput data matching some
      filter, an empty throughput proto is used at that index. All other
      throughput sources are removed. Note that this process may repeat some
      throughputs multiple times if their source name matches multiple filters.
    * it removes all basic blocks with no inverse throughput information.

  The reader is built as a wrapper over tfrecord.read_protos and has the
  following properties:
    * it can be iterated over only once,
    * the underlying IO objects are kept around until reading through all
      records in the input files.

  Args:
    input_files: The list of TFRecord files from which the samples are read.
    source_filter_list: A list of regular expressions (in text format) used to
      filter inverse throughput sources. When empty, all sources are kept.

  Returns:
    The reader that iterates over all basic blocks from the file.
  """
  proto_filters = []
  if _ACTION.value != model_options.Action.PREDICT:
    if source_filter_list:
      source_filters = [
          re.compile(source_filter) for source_filter in source_filter_list
      ]
      proto_filters.append(
          functools.partial(utils.select_throughputs, source_filters)
      )

    # When the model is trained with the random throughput selection strategy,
    # we evaluate it against the mean. For other strategies, we evaluate it
    # against the chosen strategy.
    if (
        _ACTION.value == model_options.Action.TRAIN
        or _TRAINING_THROUGHPUT_SELECTION.value
        != io_options.ThroughputSelection.RANDOM
    ):
      throughput_selection = _TRAINING_THROUGHPUT_SELECTION.value
    else:
      throughput_selection = io_options.ThroughputSelection.MEAN
    proto_filters.append(
        functools.partial(
            utils.drop_blocks_with_no_throughputs,
            _GEMATRIA_USE_SEQ2SEQ_LOSS.value,
        )
    )
    proto_filters.append(
        functools.partial(utils.aggregate_throughputs, throughput_selection)
    )
    if _INPUT_FILE_SCALING.value != 1.0:
      proto_filters.append(
          functools.partial(utils.scale_throughputs, _INPUT_FILE_SCALING.value)
      )

  protos = tfrecord.read_protos(
      input_files, throughput_pb2.BasicBlockWithThroughputProto
  )
  return utils.apply_filters(protos, proto_filters)


def _extract_basic_blocks_with_throughput(
    model: model_base.ModelBase,
    protos: Iterable[throughput_pb2.BasicBlockWithThroughputProto],
) -> Iterable[throughput.BasicBlockWithThroughput]:
  """Parses basic block data from a stream of protos.

  Removes invalid basic blocks if this is requested by command-line flags.

  Args:
    model: The Gematria model for which the basic blocks are loaded.
    protos: A stream of basic block protos with throughput. The stream will be
      iterated over only once.

  Yields:
    Parsed basic block data for the input protos. Yields one basic block with
    throughput for each proto from the input stream that passes validation by
    the model.
  """
  keep_all_blocks = not _DROP_INVALID_BLOCKS.value
  for proto in protos:
    block = throughput_protos.block_with_throughput_from_proto(proto)
    if keep_all_blocks or model.validate_basic_block_with_throughput(block):
      yield block


def _session_from_checkpoint(checkpoint_file: str) -> tf.Session:
  """Creates a local TF Session and restores it from a given checkpoint file."""
  sess = tf.Session()
  saver = tf.train.Saver()
  saver.restore(sess, checkpoint_file)
  return sess


def _task_names_from_command_line_flags() -> Sequence[str]:
  """Returns a list of task names based on the command line flags."""
  if _TASK_NAMES.value:
    return _TASK_NAMES.value
  num_filters = len(_THROUGHPUT_SOURCE_FILTERS.value)
  if num_filters <= 1:
    return ('default',)
  return tuple(f'task_{i + 1}' for i in range(num_filters))


def run_gematria_model_from_command_line_flags(
    model_class: Type[model_base.ModelBase],
    **model_kwargs: Any,
) -> None:
  """Creates and runs a Gematria model with parameters from command-line flags.

  The function depends on TensorFlow scopes to control device placement and
  other properties of the training/evaluation process. As a result, it requires
  that the model object is created inside this function, within the right scope.
  This is achieved by passing the model class and the arguments for its
  constructor rather than an initialized class object.

  Args:
    model_class: The class of the model being run. This class will be
      instantiated exactly once by the function in a TensorFlow scope configured
      by the command-line flags.
    **model_kwargs: Keyword arguments passed to model_class when an instance of
      the model is created.
  """
  np.set_printoptions(edgeitems=_GEMATRIA_NUMPY_PRINT_EDGEITEMS.value)
  if _GEMATRIA_RANDOM_SEED.value >= 0:
    tf.random.set_random_seed(_GEMATRIA_RANDOM_SEED.value)
    random.seed(_GEMATRIA_RANDOM_SEED.value)
  is_chief = _GEMATRIA_TRAINING_TASK.value == 0
  with tf.Graph().as_default():
    dev = tf.train.replica_device_setter(
        ps_tasks=_GEMATRIA_TRAINING_PS_TASKS.value
    )
    with tf.device(dev):
      with timer.scoped('Creating model: ' + model_class.__name__):
        num_replicas = _GEMATRIA_NUM_TRAINING_WORKER_REPLICAS.value
        num_replicas_to_aggregate = (
            _GEMATRIA_NUM_TRAINING_WORKER_REPLICAS_TO_AGGREGATE.value
        )
        model = model_class(  # pytype: disable=wrong-arg-types
            model_name=_MODEL_NAME.value,
            task_list=_task_names_from_command_line_flags(),
            synchronous_training=_GEMATRIA_SYNCHRONOUS_TRAINING.value,
            loss_type=_LOSS_TYPE.value,
            loss_normalization=_LOSS_NORMALIZATION.value,
            trained_variable_groups=_TRAINED_VARIABLES.value,
            learning_rate=_LEARNING_RATE.value,
            decay_steps=_DECAY_STEPS.value,
            decay_rate=_DECAY_RATE.value,
            learning_rate_schedule=_LEARNING_RATE_SCHEDULE.value,
            optimizer_type=_OPTIMIZER_TYPE.value,
            grad_clip_norm=_GRAD_CLIP_NORM.value,
            use_delta_loss=_GEMATRIA_USE_SEQ2SEQ_LOSS.value,
            collected_percentile_ranks=tuple(
                map(int, _COLLECTED_PERCENTILE_RANKS.value)
            ),
            num_training_worker_replicas=num_replicas,
            num_training_worker_replicas_to_aggregate=num_replicas_to_aggregate,
            is_chief=is_chief,
            **model_kwargs,
        )
        model.initialize()
      with timer.scoped('Loading basic blocks'):
        if _ACTION.value not in model_options.ACTIONS_WITHOUT_INPUT_DATA:
          input_files = _INPUT_FILES.value
          if not input_files:
            sys.exit(
                'At least one .tfrecord file must be specified through'
                ' --gematria_input_file.'
            )
          basic_block_protos = _make_basic_block_reader_from_command_line_flags(
              input_files, _THROUGHPUT_SOURCE_FILTERS.value
          )
          blocks_with_throughput = _extract_basic_blocks_with_throughput(
              model, basic_block_protos
          )
        else:
          basic_block_protos = None
          blocks_with_throughput = None
      max_instructions_in_batch = _GEMATRIA_MAX_INSTRUCTIONS_IN_BATCH.value
      if _ACTION.value == model_options.Action.EVAL:
        session_hooks = None
        model.run_continuous_evaluation(
            tuple(blocks_with_throughput),
            _CHECKPOINT_DIR.value,
            _GEMATRIA_SUMMARY_DIR.value,
            tf_master=_MASTER.value,
            session_hooks=session_hooks,
            eval_interval_seconds=_GEMATRIA_EVAL_INTERVAL_SECS.value,
            max_blocks_in_batch=_GEMATRIA_MAX_BLOCKS_IN_BATCH.value,
            max_instructions_in_batch=max_instructions_in_batch,
        )
      elif _ACTION.value == model_options.Action.PREDICT:
        with _session_from_checkpoint(_CHECKPOINT_FILE.value) as sess:
          output_blocks = inference.predict_for_protos(
              model,
              sess,
              basic_block_protos,
              max_blocks_in_batch=_GEMATRIA_MAX_BLOCKS_IN_BATCH.value,
              max_instructions_in_batch=max_instructions_in_batch,
          )
          tfrecord.write_protos(_GEMATRIA_OUTPUT_FILE.value, output_blocks)
      elif _ACTION.value == model_options.Action.EXPORT_GRAPH_DEF:
        graph_def = tf.get_default_graph().as_graph_def()
        graph_def = tf.compat.v1.graph_util.remove_training_nodes(
            graph_def, protected_nodes=model.output_tensor_names
        )
        if _CHECKPOINT_FILE.value:
          # When a checkpoint file is specified, replace tf.Variable nodes with
          # tf.constant() nodes with the values of the variables from this
          # checkpoint.
          with _session_from_checkpoint(_CHECKPOINT_FILE.value) as sess:
            graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=graph_def,
                output_node_names=model.output_tensor_names,
            )
        tf.io.write_graph(
            graph_def,
            logdir=os.path.dirname(_GRAPH_DEF_FILE.value),
            name=os.path.basename(_GRAPH_DEF_FILE.value),
        )
      elif _ACTION.value == model_options.Action.TRAIN:
        if is_chief:
          _resume_previous_experiment_if_needed()
        with timer.scoped('Create training session'):
          session = _monitored_training_session_from_flags(model, is_chief)
        with timer.scoped('Running the training'):
          with session:
            randomize_expected_outputs = (
                _TRAINING_THROUGHPUT_SELECTION.value
                == io_options.ThroughputSelection.RANDOM
            )
            model.train(
                session,
                tuple(blocks_with_throughput),
                max_blocks_in_batch=_GEMATRIA_MAX_BLOCKS_IN_BATCH.value,
                max_instructions_in_batch=max_instructions_in_batch,
                num_epochs=_GEMATRIA_TRAINING_NUM_EPOCHS.value,
                randomize_batches=_GEMATRIA_TRAINING_RANDOMIZE_BATCHES.value,
                randomize_expected_outputs=randomize_expected_outputs,
            )
