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

import copy
import functools
from os import path
import re
from unittest import mock

from absl import flags
from absl.testing import flagsaver
import numpy as np
import tensorflow.compat.v1 as tf

from gematria.basic_block.python import throughput_protos
from gematria.io.python import options as io_options
from gematria.io.python import tfrecord
from gematria.io.python import utils
from gematria.model.python import inference
from gematria.model.python import main_function
from gematria.model.python import model_base
from gematria.model.python import options as model_options
from gematria.proto import throughput_pb2
from gematria.testing.python import matchers
from gematria.testing.python import model_test

FLAGS = flags.FLAGS


class TestModel(model_base.ModelBase):
  """A simple model used in the tests.

  The model returns the value of a TensorFlow variable for each basic block.
  This is simple emough that the output of the model can be controlled and
  checked, but it also allows training of the model as the output actually
  depends on a trainable variable.
  """

  num_blocks_in_batch: int = 0
  num_instructions_in_batch: int = 0

  # @Override
  def _create_tf_graph(self):
    self.prediction_var = tf.get_variable(
        'prediction',
        (1, self.num_tasks),
        dtype=self.dtype,
        initializer=tf.initializers.constant(0),
    )
    self.output_shape_tensor = tf.placeholder(dtype=tf.dtypes.int32, shape=(2,))
    self.output_deltas_shape_tensor = tf.placeholder(
        dtype=tf.dtypes.int32, shape=(2,))
    if self._use_deltas:
      self._output_tensor_deltas = tf.broadcast_to(
          self.prediction_var,
          self.output_deltas_shape_tensor,
          name=self.OUTPUT_TENSOR_DELTAS_NAME,
      )
    else:
      self._output_tensor = tf.broadcast_to(
          self.prediction_var,
          self.output_shape_tensor,
          name=self.OUTPUT_TENSOR_NAME,
      )

  # @Override
  def _start_batch(self):
    super()._start_batch()
    self.num_blocks_in_batch = 0
    self.num_instructions_in_batch = 0

  # @Override
  def _add_basic_block_to_batch(self, block):
    self.num_blocks_in_batch += 1
    self.num_instructions_in_batch += len(block.instructions)

  # @Override
  def _make_batch_feed_dict(self):
    return {
        self.output_shape_tensor:
            np.array((self.num_blocks_in_batch, self.num_tasks)),
        self.output_deltas_shape_tensor:
            np.array((self.num_instructions_in_batch, self.num_tasks)),
    }

  # @Override()
  def _make_model_name(self):
    return 'TestModel'


class GematriaMainFunctionTest(model_test.TestCase):

  def setUp(self):
    self.num_blocks = 5
    super().setUp()

    # Create a .tfrecord file with basic blocks from test data.
    self.work_directory = self.create_tempdir()

    self.input_filename = path.join(self.work_directory.full_path,
                                    'input.tfrecord')
    tfrecord.write_protos(self.input_filename, self.block_protos)

  def _create_checkpoint_file(
      self,
      filename,
      prediction_value,
      *model_args,
      global_step=None,
      **model_kwargs,
  ):
    """Creates a checkpoint file for the test model.

    The checkpoint file will contain variable values set in such way that the
    model predicts 'prediction_value' for all basic blocks.

    Args:
      filename: The name of the checkpoint file.
      prediction_value: The value predicted by the model loaded from the
        checkpoint file.
      *model_args: Extra positional arguments, passed to the constructor of the
        model.
      global_step: The value of global step used for the checkpoint. When None,
        the checkpoint in the model is not modified.
      **model_kwargs: Extra keyword arguments, passed to the constructor of the
        model.
    """
    model = TestModel(*model_args, dtype=tf.dtypes.float32, **model_kwargs)
    model.initialize()
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.assign(model.prediction_var, [[prediction_value]]))
      if global_step is not None:
        sess.run(tf.assign(model.global_step, global_step))
      saver = tf.train.Saver()
      saver.save(sess, filename, global_step=global_step)

  def _assert_file_exists(self, pattern):
    """Checks that the working directory contains a file.

    Asserts that there is at least one file in the working directory matching
    the given pattern (via tf.io.gfile.glob).

    Args:
      pattern: The matched pattern.
    """
    full_path_pattern = path.join(self.work_directory.full_path, pattern)
    all_files = tf.io.gfile.listdir(self.work_directory.full_path)
    self.assertNotEmpty(
        tf.io.gfile.glob(full_path_pattern),
        'File was not found in the working directory: {}. All files: {}'.format(
            full_path_pattern, all_files),
    )

  @flagsaver.flagsaver
  def test_eval(self):
    """Tests the eval action of the Gematria main function.

    The evaluation runs in an infinite loop and updates the TensorFlow event
    file, which makes it difficult without digging deep into TensorFlow data
    structures and files. Instead of doing that, we just check that the main
    function calls the appropriate method of the file.
    """
    checkpoint_dir = '/checkpoint/dir'
    summary_dir = '/summary/dir'
    master = 'local'
    eval_interval_seconds = 987
    max_blocks_in_batch = 15
    max_instructions_in_batch = 124
    collected_percentile_ranks = [10, 20, 30, 90]

    model = None

    # The model used for training is created inside the Gematria main function.
    # To extract it, we wrap TestModel with a function that creates an instance,
    # sets up the necessary mocks on its methods, and stores the created model
    # in trained_model for inspection.
    def MockModel(*args, **kwargs):
      nonlocal model
      model = TestModel(*args, **kwargs)
      model.run_continuous_evaluation = mock.MagicMock()
      return model

    FLAGS.gematria_action = model_options.Action.EVAL
    FLAGS.gematria_input_file = (self.input_filename,)
    FLAGS.gematria_checkpoint_dir = checkpoint_dir
    FLAGS.gematria_summary_dir = summary_dir
    FLAGS.master = master
    FLAGS.gematria_eval_interval_secs = eval_interval_seconds
    FLAGS.gematria_collected_percentile_ranks = collected_percentile_ranks
    FLAGS.gematria_loss_type = model_options.LossType.HUBER
    FLAGS.gematria_loss_normalization = (
        model_options.ErrorNormalization.PERCENTAGE_ERROR)
    FLAGS.gematria_max_blocks_in_batch = max_blocks_in_batch
    FLAGS.gematria_max_instructions_in_batch = max_instructions_in_batch

    main_function.run_gematria_model_from_command_line_flags(
        MockModel, dtype=tf.dtypes.float32)

    self.assertEqual(
        model.loss_normalization,
        model_options.ErrorNormalization.PERCENTAGE_ERROR,
    )
    self.assertSequenceEqual(model.collected_percentile_ranks,
                             collected_percentile_ranks)
    self.assertEqual(model.loss_type, model_options.LossType.HUBER)

    block_filters = (
        functools.partial(utils.select_throughputs, (re.compile('.*'),)),
        functools.partial(utils.drop_blocks_with_no_throughputs, False),
        functools.partial(utils.aggregate_throughputs,
                          io_options.ThroughputSelection.MEAN),
    )
    block_protos = utils.apply_filters(self.block_protos, block_filters)
    expected_blocks = [
        throughput_protos.block_with_throughput_from_proto(block_proto)
        for block_proto in block_protos
    ]

    model.run_continuous_evaluation.assert_called_once_with(
        matchers.SequenceEqual(expected_blocks),
        checkpoint_dir,
        summary_dir,
        tf_master=master,
        session_hooks=None,
        eval_interval_seconds=eval_interval_seconds,
        max_blocks_in_batch=max_blocks_in_batch,
        max_instructions_in_batch=max_instructions_in_batch,
    )

  @flagsaver.flagsaver
  def test_predict(self):
    """Tests the predict action of the Gematria main function.

    The test sets up an input file and a checkpoint that makes the model predict
    a given value for each basic block, then runs the actual prediction code on
    this setup and checks the output.
    """
    predicted_value = 123456
    max_blocks_in_batch = 15
    max_instructions_in_batch = 124
    checkpoint_filename = path.join(self.work_directory.full_path,
                                    'checkpoint.ckpt')
    output_filename = path.join(self.work_directory.full_path,
                                'output.tfrecord')
    self._create_checkpoint_file(checkpoint_filename, predicted_value)

    model = None

    def MockModel(*args, **kwargs):
      nonlocal model
      model = TestModel(*args, **kwargs)
      return model

    # Record calls to inference.predict_for_protos() but still call the original
    # function.
    self.enter_context(
        mock.patch.object(
            inference,
            'predict_for_protos',
            side_effect=inference.predict_for_protos,
        ))

    FLAGS.gematria_action = model_options.Action.PREDICT
    FLAGS.gematria_checkpoint_file = checkpoint_filename
    FLAGS.gematria_input_file = (self.input_filename,)
    FLAGS.gematria_output_file = output_filename
    # Note that throughput source filters are ignored when the action is
    # 'predict'.
    FLAGS.gematria_throughput_source_filter = ['test:.*']
    FLAGS.gematria_max_blocks_in_batch = max_blocks_in_batch
    FLAGS.gematria_max_instructions_in_batch = max_instructions_in_batch
    main_function.run_gematria_model_from_command_line_flags(
        MockModel, dtype=tf.dtypes.float32)

    inference.predict_for_protos.assert_called_once_with(
        model,
        mock.ANY,  # The TF session.
        mock.ANY,  # An iterable object reading the basic blocks.
        max_blocks_in_batch=max_blocks_in_batch,
        max_instructions_in_batch=max_instructions_in_batch,
    )

    output_blocks = list(
        tfrecord.read_protos((output_filename,),
                             throughput_pb2.BasicBlockWithThroughputProto))
    for block in output_blocks:
      for throughput in block.inverse_throughputs:
        self.assertIn(
            throughput.source,
            [
                'TestModel, task=default',
                'llvm_sim: triple=x86_64-linux-gnu, cpu=haswell, cpu_features=',
                'test: made up values',
                'test: predicted value for tests',
            ],
        )
    expected_output_blocks = copy.deepcopy(self.block_protos)
    for expected_output_block in expected_output_blocks:
      expected_output_block.inverse_throughputs.append(
          throughput_pb2.ThroughputWithSourceProto(
              source='TestModel, task=default',
              inverse_throughput_cycles=[predicted_value],
          ))
    self.assertSequenceEqual(output_blocks, expected_output_blocks)

  @flagsaver.flagsaver
  def test_predict_with_custom_name(self):
    """Tests the predict action with a custom model name.

    The test sets up an input file and a checkpoint that makes the model predict
    a given value for each basic block, then runs the actual prediction code on
    this setup and checks the output.
    """
    predicted_value = 123456
    max_blocks_in_batch = 15
    max_instructions_in_batch = 124
    checkpoint_filename = path.join(self.work_directory.full_path,
                                    'checkpoint.ckpt')
    output_filename = path.join(self.work_directory.full_path,
                                'output.tfrecord')
    self._create_checkpoint_file(checkpoint_filename, predicted_value)

    FLAGS.gematria_action = model_options.Action.PREDICT
    FLAGS.gematria_model_name = 'CustomModelName'
    FLAGS.gematria_checkpoint_file = checkpoint_filename
    FLAGS.gematria_input_file = (self.input_filename,)
    FLAGS.gematria_output_file = output_filename
    # Note that throughput source filters are ignored when the action is
    # 'predict'.
    FLAGS.gematria_throughput_source_filter = ['test:.*']
    FLAGS.gematria_task_names = ['test_task']
    FLAGS.gematria_max_blocks_in_batch = max_blocks_in_batch
    FLAGS.gematria_max_instructions_in_batch = max_instructions_in_batch
    main_function.run_gematria_model_from_command_line_flags(
        TestModel, dtype=tf.dtypes.float32)

    output_blocks = list(
        tfrecord.read_protos((output_filename,),
                             throughput_pb2.BasicBlockWithThroughputProto))
    for block in output_blocks:
      for throughput in block.inverse_throughputs:
        self.assertIn(
            throughput.source,
            [
                'CustomModelName, task=test_task',
                'llvm_sim: triple=x86_64-linux-gnu, cpu=haswell, cpu_features=',
                'test: made up values',
                'test: predicted value for tests',
            ],
        )
    expected_output_blocks = copy.deepcopy(self.block_protos)
    for expected_output_block in expected_output_blocks:
      expected_output_block.inverse_throughputs.append(
          throughput_pb2.ThroughputWithSourceProto(
              source='CustomModelName, task=test_task',
              inverse_throughput_cycles=[predicted_value],
          ))
    self.assertSequenceEqual(output_blocks, expected_output_blocks)

  @flagsaver.flagsaver
  def test_train(self):
    """Tests the train action of the Gematria main function.

    The tests prepares training data and runs the actual training for a small
    number of epochs. Then checks that all the expected checkpoint and summary
    files were created in the provided checkpoint directory, the checkpoint can
    be restored, and that it contains expected values.
    """
    num_epochs = 10
    max_blocks_in_batch = 15
    max_instructions_in_batch = 124
    learning_rate = 0.321
    randomize_batches = False
    training_throughput_selection = io_options.ThroughputSelection.RANDOM
    checkpoint_dir = path.join(self.work_directory.full_path, 'checkpoint')
    use_seq2seq_loss = False  # The default is True.

    model = None

    def MockModel(*args, **kwargs):
      nonlocal model
      self.assertEqual(kwargs['learning_rate'], learning_rate)
      model = TestModel(*args, **kwargs)
      # Record calls to model.train(), but still call the original method.
      mock_train = mock.MagicMock(side_effect=model.train)
      model.train = mock_train
      return model

    FLAGS.gematria_action = model_options.Action.TRAIN
    FLAGS.gematria_input_file = (self.input_filename,)
    FLAGS.gematria_checkpoint_dir = checkpoint_dir
    FLAGS.gematria_summary_dir = path.join(self.work_directory.full_path,
                                           'summary')
    FLAGS.gematria_training_num_epochs = num_epochs
    FLAGS.gematria_training_randomize_batches = randomize_batches
    FLAGS.gematria_max_blocks_in_batch = max_blocks_in_batch
    FLAGS.gematria_max_instructions_in_batch = max_instructions_in_batch
    FLAGS.gematria_use_seq2seq_loss = use_seq2seq_loss
    FLAGS.gematria_learning_rate = learning_rate
    FLAGS.gematria_training_throughput_selection = training_throughput_selection

    main_function.run_gematria_model_from_command_line_flags(
        MockModel, dtype=tf.dtypes.float32)

    block_filters = (
        functools.partial(utils.select_throughputs, (re.compile('.*'),)),
        functools.partial(utils.drop_blocks_with_no_throughputs, False),
        functools.partial(utils.aggregate_throughputs,
                          training_throughput_selection),
    )
    block_protos = utils.apply_filters(self.block_protos, block_filters)
    expected_blocks = [
        throughput_protos.block_with_throughput_from_proto(block_proto)
        for block_proto in block_protos
    ]

    self.assertEqual(model.num_tasks, 1)
    self.assertEqual(model._use_delta_loss, use_seq2seq_loss)
    model.train.assert_called_once_with(
        mock.ANY,  # The TF session.
        matchers.SequenceEqual(expected_blocks),
        max_blocks_in_batch=max_blocks_in_batch,
        max_instructions_in_batch=max_instructions_in_batch,
        num_epochs=num_epochs,
        randomize_batches=randomize_batches,
        randomize_expected_outputs=True,
    )

    # Check that the files created by the monitored session are there.
    self._assert_file_exists('checkpoint/checkpoint')
    self._assert_file_exists('checkpoint/graph.pbtxt')
    self._assert_file_exists('checkpoint/model.ckpt-*')
    self._assert_file_exists('summary/events.out.tfevents.*')

    # Try to load the latest checkpoint with the model.
    model = TestModel(dtype=tf.dtypes.float32)
    model.initialize()
    with self.session() as sess:
      saver = tf.train.Saver()
      latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
      saver.restore(sess, latest_checkpoint)

      # Inspect the value of the prediction variable. It is initialized to zero,
      # and it must change during the training. While it is not clear what the
      # actual value will be, it is certain that it will be greater than zero.
      prediction = sess.run(model.prediction_var)
      self.assertLen(prediction, 1)
      self.assertGreater(prediction[0], 0)

      # Check the value of the global step loaded from the checkpoint. This
      # should be equal to the number of training epochs.
      self.assertEqual(sess.run(model.global_step), num_epochs)

  @flagsaver.flagsaver
  def test_train_with_min_throughput(self):
    """Tests the train action of the Gematria main function.

    This test is a lightweight version of test_train but uses a different
    throughput selection strategy and only checks that the right call to
    model.train() was made, but does not check the actual training and all its
    artifacts.
    """
    num_epochs = 10
    max_blocks_in_batch = 15
    max_instructions_in_batch = 124
    randomize_batches = False
    training_throughput_selection = io_options.ThroughputSelection.MIN

    model = None

    def MockModel(*args, **kwargs):
      nonlocal model
      model = TestModel(*args, **kwargs)
      model.train = mock.MagicMock()
      return model

    FLAGS.gematria_action = model_options.Action.TRAIN
    FLAGS.gematria_input_file = (self.input_filename,)
    FLAGS.gematria_summary_dir = path.join(self.work_directory.full_path,
                                           'summary')
    FLAGS.gematria_training_num_epochs = num_epochs
    FLAGS.gematria_training_randomize_batches = randomize_batches
    FLAGS.gematria_max_blocks_in_batch = max_blocks_in_batch
    FLAGS.gematria_max_instructions_in_batch = max_instructions_in_batch
    FLAGS.gematria_training_throughput_selection = training_throughput_selection

    main_function.run_gematria_model_from_command_line_flags(
        MockModel, dtype=tf.dtypes.float32)

    block_filters = (
        functools.partial(utils.select_throughputs, (re.compile('.*'),)),
        functools.partial(utils.drop_blocks_with_no_throughputs, False),
        functools.partial(utils.aggregate_throughputs,
                          training_throughput_selection),
    )
    block_protos = utils.apply_filters(self.block_protos, block_filters)
    expected_blocks = [
        throughput_protos.block_with_throughput_from_proto(block_proto)
        for block_proto in block_protos
    ]

    model.train.assert_called_once_with(
        mock.ANY,  # The TF session.
        matchers.SequenceEqual(expected_blocks),
        max_blocks_in_batch=max_blocks_in_batch,
        max_instructions_in_batch=max_instructions_in_batch,
        num_epochs=num_epochs,
        randomize_batches=randomize_batches,
        randomize_expected_outputs=False,
    )

  @flagsaver.flagsaver
  def test_train_with_min_throughput_scaled(self):
    """Tests the train action of the Gematria main function.

    This test is a lightweight version of test_train but uses a different
    throughput selection strategy and only checks that the right call to
    model.train() was made, but does not check the actual training and all its
    artifacts.
    """
    num_epochs = 10
    max_blocks_in_batch = 15
    max_instructions_in_batch = 124
    randomize_batches = False
    training_throughput_selection = io_options.ThroughputSelection.MIN
    throughput_scaling_factor = 3.0

    model = None

    def MockModel(*args, **kwargs):
      nonlocal model
      model = TestModel(*args, **kwargs)
      model.train = mock.MagicMock()
      return model

    FLAGS.gematria_action = model_options.Action.TRAIN
    FLAGS.gematria_input_file = (self.input_filename,)
    FLAGS.gematria_summary_dir = path.join(self.work_directory.full_path,
                                           'summary')
    FLAGS.gematria_training_num_epochs = num_epochs
    FLAGS.gematria_training_randomize_batches = randomize_batches
    FLAGS.gematria_max_blocks_in_batch = max_blocks_in_batch
    FLAGS.gematria_max_instructions_in_batch = max_instructions_in_batch
    FLAGS.gematria_training_throughput_selection = training_throughput_selection
    FLAGS.gematria_input_file_scaling = throughput_scaling_factor

    main_function.run_gematria_model_from_command_line_flags(
        MockModel, dtype=tf.dtypes.float32)

    block_filters = (
        functools.partial(utils.select_throughputs, (re.compile('.*'),)),
        functools.partial(utils.drop_blocks_with_no_throughputs, False),
        functools.partial(utils.aggregate_throughputs,
                          training_throughput_selection),
        functools.partial(utils.scale_throughputs, throughput_scaling_factor),
    )
    block_protos = utils.apply_filters(self.block_protos, block_filters)
    expected_blocks = [
        throughput_protos.block_with_throughput_from_proto(block_proto)
        for block_proto in block_protos
    ]

    model.train.assert_called_once_with(
        mock.ANY,  # The TF session.
        matchers.SequenceEqual(expected_blocks),
        max_blocks_in_batch=max_blocks_in_batch,
        max_instructions_in_batch=max_instructions_in_batch,
        num_epochs=num_epochs,
        randomize_batches=randomize_batches,
        randomize_expected_outputs=False,
    )

  def test_train_with_resume(self):
    """Tests the train action of the Gematria main function.

    This test also uses the resume function that copies data from a previous
    experiment to the directory of the current experiment.
    """
    predicted_value = 123.0
    global_step = 999

    old_checkpoint_dir = path.join(self.work_directory, 'old')
    new_checkpoint_dir = path.join(self.work_directory, 'new')
    summary_dir = path.join(self.work_directory, 'summaries')

    # Create a checkpoint in the "old" directory.
    old_checkpoint_file = path.join(old_checkpoint_dir, 'model.ckpt')
    tf.io.gfile.makedirs(old_checkpoint_dir)
    self._create_checkpoint_file(
        old_checkpoint_file, predicted_value, global_step=global_step)

    # Check that the checkpoint dir has the expected structure. There must be at
    # least a file called "checkpoint" that contains the list of the actual
    # checkpoints in text format. We check that the file is there, it contains
    # references to the old dir and no references to the "new" checkpoint dir.
    with tf.io.gfile.GFile(path.join(old_checkpoint_dir, 'checkpoint'),
                           'r') as f:
      checkpoint_list_pbtxt = f.read()
      self.assertIn(old_checkpoint_file, checkpoint_list_pbtxt)
      self.assertNotIn(new_checkpoint_dir, checkpoint_list_pbtxt)
    checkpoint_files = tf.io.gfile.glob(old_checkpoint_file + '*')
    self.assertNotEmpty(checkpoint_files)

    model = None

    def MockModel(*args, **kwargs):
      nonlocal model
      model = TestModel(*args, **kwargs)
      model.train = mock.MagicMock()
      return model

    # NOTE(ondrasej): When invoked this way, flagsaver sets the values of all
    # flags at once, triggers the validation only once at the end, and avoids
    # false error reports from incomplete flag assignments.
    with flagsaver.flagsaver(
        gematria_action=model_options.Action.TRAIN,
        gematria_input_file=(self.input_filename,),
        gematria_checkpoint_dir=new_checkpoint_dir,
        gematria_summary_dir=summary_dir,
        gematria_resume_from_dir=old_checkpoint_dir,
        gematria_resume_to_dir=new_checkpoint_dir,
    ):
      main_function.run_gematria_model_from_command_line_flags(
          MockModel, dtype=tf.dtypes.float32)

    self.assertTrue(tf.io.gfile.exists(new_checkpoint_dir))
    old_glob = {
        path.basename(filename)
        for filename in tf.io.gfile.glob(path.join(old_checkpoint_dir, '*'))
    }
    new_glob = {
        path.basename(filename)
        for filename in tf.io.gfile.glob(path.join(new_checkpoint_dir, '*'))
    }

    # Check that the new directory contains all files from the old one. There
    # may be extra files created by RunGematriaModelFromCommandLineFlags(), such
    # as events.out.tfevent.* files and additional checkpoints.
    self.assertTrue(
        old_glob.issubset(new_glob),
        f'Not all files were copied.\nOld: {old_glob}\nNew: {new_glob}',
    )

    # Check that all paths related to the old directory have been replaced with
    # the new one.
    with tf.io.gfile.GFile(path.join(new_checkpoint_dir, 'checkpoint'),
                           'r') as f:
      checkpoint_list_pbtxt = f.read()
      self.assertNotIn(old_checkpoint_dir, checkpoint_list_pbtxt)
      self.assertIn(new_checkpoint_dir, checkpoint_list_pbtxt)

  @flagsaver.flagsaver
  def test_eval_with_source_filters(self):
    """Tests filtering basic blocks by source filters.

    Loads the first five basic blocks from testdata with filters that are
    matched by only four of them. Verifies that the correct number of basic
    blocks is loaded.
    """
    model = None

    def MockModel(*args, **kwargs):
      nonlocal model
      model = TestModel(*args, **kwargs)
      model.run_continuous_evaluation = mock.MagicMock()
      return model

    FLAGS.gematria_action = model_options.Action.EVAL
    FLAGS.gematria_input_file = (self.input_filename,)
    FLAGS.gematria_throughput_source_filter = [
        'llvm_sim: .*',
        'test: made up values',
    ]

    main_function.run_gematria_model_from_command_line_flags(
        MockModel, dtype=tf.dtypes.float32)

    calls = model.run_continuous_evaluation.call_args_list
    self.assertLen(calls, 1)
    # Extract the first positional argument of the first call. This is an
    # iterable object that contains the basic blocks used for evaluation.
    #                  +---------- First call in call_args_list.
    #                  |  +------- Positional arguments.
    #                  |  |  +---- First argument
    #                  v  v  v
    block_list = calls[0][0][0]
    self.assertLen(block_list, 5)

  @flagsaver.flagsaver
  def test_eval_with_multiple_tasks(self):
    """Tests training a multi-task model.

    This test is much simpler than the model training test above, and it checks
    mainly that the multi-task command-line flags are passed to the model as
    expected.
    """
    model = None

    def MockModel(*args, **kwargs):
      nonlocal model
      model = TestModel(*args, **kwargs)
      model.run_continuous_evaluation = mock.MagicMock()
      return model

    FLAGS.gematria_action = 'eval'
    FLAGS.gematria_input_file = (self.input_filename,)
    FLAGS.gematria_throughput_source_filter = ['hsw', 'skx', 'icx']

    main_function.run_gematria_model_from_command_line_flags(
        MockModel, dtype=tf.dtypes.float32)

    self.assertLen(FLAGS.gematria_throughput_source_filter, model.num_tasks)
    self.assertEqual(model.task_list, ('task_1', 'task_2', 'task_3'))

  @flagsaver.flagsaver
  def test_export_graph_def(self):
    """Tests exporting the model to a GraphDef proto."""
    graph_def_filename = path.join(self.work_directory.full_path,
                                   'graph_def.pbtxt')

    FLAGS.gematria_action = model_options.Action.EXPORT_GRAPH_DEF
    FLAGS.gematria_graph_def_file = graph_def_filename

    main_function.run_gematria_model_from_command_line_flags(
        TestModel, dtype=tf.dtypes.float32)
    with open(graph_def_filename, 'r') as graph_def_file:
      graph_def_pbtxt = graph_def_file.read()
    # We did not replace variable nodes with constants, so there should be at
    # least one variable node.
    self.assertIn('Variable', graph_def_pbtxt)

  @flagsaver.flagsaver
  def test_export_frozen_graph_def(self):
    """Tests exporting a frozen model to a GraphDef proto."""
    predicted_value = 123654
    graph_def_filename = path.join(self.work_directory.full_path,
                                   'graph_def.pbtxt')

    checkpoint_filename = path.join(self.work_directory.full_path,
                                    'checkpoint.ckpt')
    self._create_checkpoint_file(checkpoint_filename, predicted_value)

    FLAGS.gematria_action = model_options.Action.EXPORT_GRAPH_DEF
    FLAGS.gematria_graph_def_file = graph_def_filename
    FLAGS.gematria_checkpoint_file = checkpoint_filename

    main_function.run_gematria_model_from_command_line_flags(
        TestModel, dtype=tf.dtypes.float32)
    with open(graph_def_filename, 'r') as graph_def_file:
      graph_def_pbtxt = graph_def_file.read()
    # Check that the graph definition is not empty, there are no variable nodes,
    # and it contains the predicted value (which should have been injected into
    # it as a constant).
    self.assertNotEmpty(graph_def_pbtxt)
    self.assertNotIn('Variable', graph_def_pbtxt)
    self.assertIn(str(predicted_value), graph_def_pbtxt)

  @flagsaver.flagsaver
  def test_multi_task_flags(self):
    """Tests validation of multi-task learning flags."""
    # Check that the flags are valid at the beginning.
    FLAGS.validate_all_flags()

    # Check that it is OK to skip all task names.
    FLAGS.gematria_throughput_source_filter = ['foo', 'bar']
    FLAGS.gematria_task_names = []
    FLAGS.validate_all_flags()

    # Check that it is OK to supply task names.
    FLAGS.gematria_throughput_source_filter = ['hello', 'world']
    FLAGS.gematria_task_names = ['foo', 'bar']
    FLAGS.validate_all_flags()

    # Check that it is not OK if both lists are non-empty and the sizes do not
    # match.
    with self.assertRaises(flags.IllegalFlagValueError):
      FLAGS.gematria_throughput_source_filter = ['foo', 'bar', 'baz']
      FLAGS.gematria_throughput_source_filter = ['alice', 'bob']
      FLAGS.validate_all_flags()


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
