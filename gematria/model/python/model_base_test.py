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
"""Tests for the ModelBase class."""

from gematria.basic_block.python import throughput
from gematria.model.python import model_base
from gematria.model.python import options
from gematria.testing.python import model_test
import numpy as np
import tensorflow.compat.v1 as tf

# The tolerance used in tests with heavier use of float32 arithmetics.
_TOLERANCE = 1e-6


class TestModel(model_base.ModelBase):
  """A simple class inheriting from ModelBase.

  The model is fully deterministic: it keeps track of the number of visited
  basic blocks per batch, and returns predictions based on this number:
  - in the seq2num mode, it returns the number of visited basic blocks
    (including the current one) as the prediction, i.e. it returns 1 for the
    first basic block, 2 for the second one, ...
  - in the seq2seq mode, it returns the number of visited basic blocks
    (including the current one) as the prediction for each prefix of the basic
    block, i.e. it returns [1, ..., 1] for the first basic block, [2, ..., 2]
    for the second one, ...

  The model is implemented by directly using tf.placeholder() nodes as the
  output tensors, and filling them in through the returned feed dict.
  """

  def __init__(self, use_custom_output_names=False, **kwargs):
    super().__init__(**kwargs)
    self.num_visited_blocks = 0
    self.num_scheduled_instructions = 0
    self.batch_sizes = []
    self.use_custom_output_names = use_custom_output_names

  # @Override
  def _create_tf_graph(self):
    if not self._use_deltas:
      output_name = model_base.ModelBase.OUTPUT_TENSOR_NAME
      if self.use_custom_output_names:
        output_name = 'TestModel.output_tensor'
      self._output_tensor = tf.placeholder(
          self.dtype, (None, self.num_tasks), name=output_name
      )
    else:
      output_deltas_name = model_base.ModelBase.OUTPUT_TENSOR_DELTAS_NAME
      if self.use_custom_output_names:
        output_deltas_name = 'TestModel.output_tensor_deltas'
      self._output_tensor_deltas = tf.placeholder(
          self.dtype, (None, self.num_tasks), name=output_deltas_name
      )

  # @Override
  def _create_optimizer(self):
    # We can't create an optimizer: this model doesn't have any variables that
    # could be optimized, and the TF optimizers raise an exception in this
    # situation. This is not a problem - training is sufficiently tested by the
    # tests of the subclasses.
    self._decayed_learning_rate = 0.001
    pass

  # @Override
  def _make_model_name(self):
    return 'TestModel'

  # @Override
  def _add_basic_block_to_batch(self, block):
    num_instructions = len(block.instructions)
    self.num_visited_blocks += 1
    self.num_scheduled_instructions += num_instructions
    if not self._use_deltas:
      self._batch_collected_outputs.append(
          [self.num_visited_blocks + i for i in range(self.num_tasks)]
      )
    else:
      self._batch_collected_outputs.extend(
          [self.num_visited_blocks + i for i in range(self.num_tasks)]
          * num_instructions
      )

  # @Override
  def _start_batch(self):
    super()._start_batch()
    self._batch_collected_outputs = []

  # @Override
  def _make_batch_feed_dict(self):
    output_tensor = (
        self._output_tensor_deltas if self._use_deltas else self._output_tensor
    )
    return {
        output_tensor: np.array(
            self._batch_collected_outputs, dtype=self.numpy_dtype
        ).reshape((-1, self.num_tasks)),
    }

  # @Override
  def schedule_batch(self, basic_blocks, *args, **kwargs):
    self.batch_sizes.append(len(basic_blocks))
    return super().schedule_batch(basic_blocks, *args, **kwargs)


class TestModelWithVarGroups(model_base.ModelBase):
  """A model that uses variable groups.

  This model computes linear regression from the number of instructions in the
  basic block. The model uses two variable groups - one for weight variables and
  one for bias variables so that they can be trained separately.
  """

  # The names of the variable groups in the model.
  WEIGHTS = 'weights'
  BIAS = 'bias'

  def _create_tf_graph(self):
    assert not self._use_deltas, 'This model does not support seq2seq.'
    self._input_tensor = tf.placeholder(
        self.dtype, (None, 1), name='TestModelWithVarGroups._input_tensor'
    )
    output_parts = []
    # NOTE(ondrasej): The weights are initialized to 0.5, and the biases are
    # initialized to -0.5. These initial values are intentionally chosen because
    # they do not provide good predictions, and are likely to be changed by the
    # optimizer.
    for task in self.task_list:
      weight = tf.get_variable(
          name=f'weight_{task}',
          shape=(1,),
          dtype=self.dtype,
          initializer=tf.keras.initializers.constant(0.5),
      )
      self._variable_groups[TestModelWithVarGroups.WEIGHTS].append(weight)
      bias = tf.get_variable(
          name=f'bias_{task}',
          shape=(1,),
          dtype=self.dtype,
          initializer=tf.keras.initializers.constant(-0.5),
      )
      self._variable_groups[TestModelWithVarGroups.BIAS].append(bias)
      output_parts.append(weight * self._input_tensor + bias)
    self._output_tensor = tf.concat(output_parts, axis=1)

  def _make_model_name(self):
    return 'TestModelWithVarGroups'

  def _start_batch(self):
    super()._start_batch()
    self._batch_block_sizes = []

  def _add_basic_block_to_batch(self, block):
    self._batch_block_sizes.append(len(block.instructions))

  def _make_batch_feed_dict(self):
    return {
        self._input_tensor: np.array(
            self._batch_block_sizes, dtype=self.numpy_dtype
        ).reshape((-1, 1)),
    }


class ModelBaseTest(model_test.TestCase):
  """The test case for ModelBase."""

  def setUp(self):
    self.num_blocks = 10
    super().setUp()

  def test_initialize_model_base(self):
    model = TestModel(dtype=tf.dtypes.int32)
    self.assertEqual(model.dtype, tf.dtypes.int32)
    self.assertEqual(model.numpy_dtype, np.int32)
    self.assertIsNotNone(model.global_step)

    model.initialize()

  def test_output_tensor_names(self):
    for use_custom_names in [True, False]:
      with tf.Graph().as_default():
        model = TestModel(
            dtype=tf.dtypes.float32, use_custom_output_names=use_custom_names
        )
        model.initialize()
        # NOTE(ondrasej): TensorFlow adds a ":\d+" to all tensor names. The
        # number is the index of the tensor in the list of outputs of the op
        # that produced it. In case of this model, the output tensor is the
        # first output of the identity op (used for renaming the tensor).
        self.assertEqual(
            model.output_tensor.name,
            model_base.ModelBase.OUTPUT_TENSOR_NAME + ':0',
        )

        with self.assertRaisesRegex(
            AttributeError, 'output_tensor_deltas is available only'
        ):
          _ = model.output_tensor_deltas

  def test_output_tensor_names_seq2seq(self):
    for use_custom_names in [True, False]:
      with tf.Graph().as_default():
        model_seq2seq = TestModel(
            use_deltas=True,
            dtype=tf.dtypes.float32,
            use_custom_output_names=use_custom_names,
        )
        model_seq2seq.initialize()

        self.assertEqual(
            model_seq2seq.output_tensor.name,
            model_base.ModelBase.OUTPUT_TENSOR_NAME + ':0',
        )
        self.assertEqual(
            model_seq2seq.output_tensor_deltas.name,
            model_base.ModelBase.OUTPUT_TENSOR_DELTAS_NAME + ':0',
        )

  def test_schedule_batch_with_throughputs(self):
    model = TestModel(dtype=tf.dtypes.float32)
    model.initialize()

    # Schedule a batch with no limits.
    full_schedule = model.schedule_batch(self.blocks_with_throughput)
    self.assertLen(self.blocks_with_throughput, model.num_visited_blocks)
    expected_outputs = full_schedule[model._expected_outputs]
    self.assertEqual(
        expected_outputs.shape, (len(self.blocks_with_throughput), 1)
    )

    batch_size = 3
    model.num_visited_blocks = 0
    block_batch_schedule = model.schedule_batch(
        self.blocks_with_throughput, max_blocks_in_batch=batch_size
    )
    self.assertEqual(model.num_visited_blocks, batch_size)
    expected_outputs = block_batch_schedule[model._expected_outputs]
    self.assertEqual(expected_outputs.shape, (batch_size, 1))

    with self.session() as sess:
      output = sess.run(model.output_tensor, feed_dict=full_schedule)
      self.assertAllEqual(
          output, [[x + 1] for x in range(len(self.blocks_with_throughput))]
      )

      output = sess.run(model.output_tensor, feed_dict=block_batch_schedule)
      self.assertAllEqual(output, [[x + 1] for x in range(batch_size)])

  def test_schedule_batch_with_throughputs_with_deltas(self):
    model = TestModel(dtype=tf.dtypes.float32, use_deltas=True)
    model.initialize()

    # Schedule a batch with no limits.
    full_schedule = model.schedule_batch(self.blocks_with_throughput)
    self.assertLen(self.blocks_with_throughput, model.num_visited_blocks)
    expected_outputs = full_schedule[model._expected_outputs]

    self.assertEqual(
        expected_outputs.shape, (len(self.blocks_with_throughput), 1)
    )
    expected_outputs_prefixes = full_schedule[model._expected_outputs_deltas]
    expected_num_prefixes = sum(
        len(block.instructions) for block in self.blocks
    )
    self.assertEqual(
        expected_outputs_prefixes.shape, (expected_num_prefixes, 1)
    )

    batch_size = 3
    model.num_visited_blocks = 0
    block_batch_schedule = model.schedule_batch(
        self.blocks_with_throughput, max_blocks_in_batch=batch_size
    )
    self.assertEqual(model.num_visited_blocks, batch_size)
    expected_outputs = block_batch_schedule[model._expected_outputs]
    self.assertEqual(expected_outputs.shape, (batch_size, 1))

    expected_outputs_prefixes = block_batch_schedule[
        model._expected_outputs_deltas
    ]
    expected_len_prefixes = sum(
        len(block.instructions) for block in self.blocks[:batch_size]
    )
    self.assertEqual(
        expected_outputs_prefixes.shape, (expected_len_prefixes, 1)
    )

    with self.session() as sess:
      output_blocks, output_deltas = sess.run(
          [model.output_tensor, model.output_tensor_deltas],
          feed_dict=full_schedule,
      )

      expected_output_blocks = []
      expected_output_deltas = []
      for i, block in enumerate(self.blocks_with_throughput):
        expected_output_blocks.append((i + 1) * len(block.block.instructions))
        for _ in block.block.instructions:
          expected_output_deltas.append([i + 1])

      output_blocks = np.reshape(output_blocks, 10)

      self.assertAllEqual(output_blocks, expected_output_blocks)
      self.assertAllEqual(output_deltas, expected_output_deltas)

      output_blocks, output_deltas = sess.run(
          [model.output_tensor, model.output_tensor_deltas],
          feed_dict=block_batch_schedule,
      )
      output_blocks = np.reshape(output_blocks, 3)

      expected_output_blocks = []
      expected_output_deltas = []
      for i, block in enumerate(self.blocks_with_throughput[:batch_size]):
        expected_output_blocks.append((i + 1) * len(block.block.instructions))
        for _ in block.block.instructions:
          expected_output_deltas.append([i + 1])

      self.assertAllEqual(output_blocks, expected_output_blocks)
      self.assertAllEqual(output_deltas, expected_output_deltas)

  def test_schedule_batch_and_train_with_masked_outputs(self):
    task_list = ('task_1', 'task_2')
    model = TestModelWithVarGroups(
        loss_type=options.LossType.MEAN_ABSOLUTE_ERROR,
        loss_normalization=options.ErrorNormalization.PERCENTAGE_ERROR,
        dtype=tf.dtypes.float32,
        use_deltas=False,
        task_list=task_list,
        learning_rate=0.3,
    )
    model.initialize()

    # Pick two blocks in such a way that each of them has one throughput defined
    # and other throughput missing.
    blocks = self.blocks_with_throughput[:2]
    blocks[0].throughputs = (blocks[0].throughputs[0], None)
    blocks[1].throughputs = (None, blocks[1].throughputs[1])

    feed_dict = model.schedule_batch(blocks)
    self.assertAllEqual(
        feed_dict[model._output_mask], ((True, False), (False, True))
    )

    with self.session() as sess:
      self.check_training_model(
          model,
          num_epochs=30,
          blocks=blocks,
          session=sess,
          print_output_to_log=True,
      )

  def test_expected_outputs_delta(self):
    model = TestModel(dtype=tf.dtypes.float32, use_deltas=True)
    model.initialize()

    for block in self.blocks_with_throughput:
      schedule = model.schedule_batch([block], randomize_batch=False)
      expected_outputs = schedule[model._expected_outputs]
      expected_output_deltas = schedule[model._expected_outputs_deltas]

      self.assertEqual(expected_outputs.shape, (1, 1))
      self.assertEqual(
          expected_output_deltas.shape, (len(block.block.instructions), 1)
      )

      # Check that the expected (aggregate) output is what we have in the proto.
      self.assertEqual(
          expected_outputs[0], block.throughputs[0].inverse_throughput_cycles[0]
      )
      # Check that the expected deltas sum up to the expected output. Note that
      # this holds for data coming from the simulator; data coming from real
      # measurement will also contain additional difference coming from
      # measurement noise.
      self.assertNear(
          np.sum(expected_output_deltas), expected_outputs[0], _TOLERANCE
      )

  def test_randomized_expected_outputs_delta(self):
    model = TestModel(dtype=tf.dtypes.float32, use_deltas=True)
    model.initialize()

    for block in self.blocks_with_throughput:
      schedule = model.schedule_batch([block], randomize_expected_outputs=True)
      expected_outputs = schedule[model._expected_outputs]
      expected_output_deltas = schedule[model._expected_outputs_deltas]

      self.assertEqual(expected_outputs.shape, (1, 1))
      self.assertEqual(
          expected_output_deltas.shape, (len(block.block.instructions), 1)
      )

      # Check that the expected output is one of the values from the proto.
      inverse_throughputs = block.throughputs[0]
      self.assertIn(
          expected_outputs[0], inverse_throughputs.inverse_throughput_cycles
      )
      # Check that the expected deltas sum up to the expected output. In the
      # randomized mode, the expected output is a difference between a randomly
      # chosen throughput for the current prefix and a randomly chosen
      # throughput for the previous throughput. To check that the delta is
      # correct, we build the set of all possible deltas, and check that the
      # expected value is in this set.
      previous_prefix_throughputs = [0.0]
      for i, prefix_throughputs in enumerate(
          inverse_throughputs.prefix_inverse_throughput_cycles
      ):
        expected_output_delta = float(expected_output_deltas[i])
        possible_deltas = set()
        for prefix_throughput in prefix_throughputs:
          for previous_prefix_throughput in previous_prefix_throughputs:
            possible_deltas.add(prefix_throughput - previous_prefix_throughput)
        previous_prefix_throughputs = prefix_throughputs
        self.assertIn(expected_output_delta, possible_deltas)

  def test_schedule_batch_with_instruction_limit(self):
    model = TestModel(dtype=tf.dtypes.float32)
    model.initialize()

    # For batches limited by the number of instructions, we only check that the
    # limits were respected; we do not evaluate them.
    max_instructions = 10
    num_tests = 15
    for _ in range(num_tests):
      model.num_visited_blocks = 0
      model.num_scheduled_instructions = 0
      model.schedule_batch(
          self.blocks_with_throughput,
          max_instructions_in_batch=max_instructions,
          randomize_batch=True,
      )
      self.assertLessEqual(model.num_scheduled_instructions, max_instructions)
      self.assertGreater(model.num_visited_blocks, 0)

    model = TestModel(
        dtype=tf.dtypes.float32, loss_type=options.LossType.MEAN_SQUARED_ERROR
    )
    model.initialize()

    with self.session() as sess:
      # Use the second block from testdata/basic_blocks_with_throughput.pbtxt.
      # This basic block has overall inverse throughput equal to 2.0, i.e. the
      # loss with MSE and absolute error must be 1.0.
      schedule = model.schedule_batch([self.blocks_with_throughput[1]])
      loss = sess.run(model.loss_tensor, feed_dict=schedule)
      self.assertEqual(loss, 1.0)

  def test_seq2seq_delta_loss(self):
    model = TestModel(
        dtype=tf.dtypes.float32,
        loss_type=options.LossType.MEAN_SQUARED_ERROR,
        use_deltas=True,
        use_delta_loss=True,
    )
    model.initialize()

    with self.session() as sess:
      # Use the second block from testdata/basic_blocks_with_throughput.pbtxt.
      # This basic block has 5 prefixes with inverse throughputs
      # [1, 1, 1, 1, 2] and deltas [1, 0, 0, 0, 1]. The model predicts
      # deltas [1, 1, 1, 1, 1] and the delta-based loss is thus
      # (0 + 1 + 1 + 1 + 0)/5.
      schedule = model.schedule_batch([self.blocks_with_throughput[1]])
      loss = sess.run(model.loss_tensor, feed_dict=schedule)
      self.assertNear(loss, (0 + 1 + 1 + 1 + 0) / 5, 1e-6)

  def test_seq2seq_no_delta_loss(self):
    model = TestModel(
        dtype=tf.dtypes.float32,
        loss_type=options.LossType.MEAN_SQUARED_ERROR,
        use_deltas=True,
        use_delta_loss=False,
    )
    model.initialize()

    with self.session() as sess:
      # Use the second block from testdata/basic_blocks_with_throughput.pbtxt.
      # This basic block has 5 prefixes with inverse throughputs
      # [1, 1, 1, 4/3, 2] and deltas [1, 0, 0, 1/3, 2/3]. The model predicts
      # deltas [1, 1, 1, 1, 1] and the per-basic block loss is thus (5-2)^2 = 9.
      schedule = model.schedule_batch([self.blocks_with_throughput[1]])
      loss = sess.run(model.loss_tensor, feed_dict=schedule)
      self.assertNear(loss, 9, 1e-6)

  def check_predict(
      self,
      model,
      max_blocks_in_batch,
      max_instructions_in_batch,
      expected_batch_sizes,
  ):
    """Checks the prediction of the test model with the given batch size.

    Uses the instrumentation in model.schedule_batch() to verify that the input
    sequence was partitioned as requested.

    Args:
      model: The model used in tests.
      max_blocks_in_batch: The maximal number of basic blocks in a batch, passed
        to model.predict().
      max_instructions_in_batch: The maximal number of instructions in a batch,
        passed to model.predict().
      expected_batch_sizes: A collection of expected sizes of batches processed
        by model.predict(), verified by this method.
    """
    with self.session() as sess:
      output_blocks = tuple(
          model.predict(
              sess,
              self.blocks,
              max_blocks_in_batch=max_blocks_in_batch,
              max_instructions_in_batch=max_instructions_in_batch,
          )
      )
      self.assertEqual(model.batch_sizes, expected_batch_sizes)
      self.assertLen(output_blocks, len(self.blocks_with_throughput))
      for index, (in_block, out_block) in enumerate(
          zip(self.blocks, output_blocks)
      ):
        # The prediction of the model is the number of calls to
        # model._add_basic_block_to_batch(). There is one call per basic block,
        # so we can get the expected value from the index of the basic block.
        expected_inverse_throughputs = []
        for task_index in range(model.num_tasks):
          expected_inverse_throughputs.append((index + 1 + task_index,))
        self.assertEqual(in_block, out_block.block)
        self.assertLen(out_block.throughputs, model.num_tasks)
        predicted_throughputs = [
            throughput.inverse_throughput_cycles
            for throughput in out_block.throughputs
        ]
        self.assertSequenceEqual(
            predicted_throughputs, expected_inverse_throughputs
        )

  def test_predict_single_batch(self):
    model = TestModel(dtype=tf.dtypes.float32)
    model.initialize()

    self.check_predict(model, None, None, [len(self.blocks_with_throughput)])

  def test_predict_multiple_batches(self):
    model = TestModel(dtype=tf.dtypes.float32)
    model.initialize()

    batch_size = 3
    num_blocks = len(self.blocks_with_throughput)
    expected_batch_sizes = [batch_size] * (num_blocks // batch_size) + [
        num_blocks % batch_size
    ]
    self.check_predict(model, 3, None, expected_batch_sizes)

  def test_predict_with_instruction_limit(self):
    model = TestModel(dtype=tf.dtypes.float32)
    model.initialize()

    max_instructions_in_batch = 10
    # Lengths of blocks in self.blocks_with_throughput are:
    # [1, 5, 1, 8, 3, 4, 1, 2, 9, 4].
    expected_batch_sizes = [3, 1, 4, 1, 1]
    self.check_predict(
        model, None, max_instructions_in_batch, expected_batch_sizes
    )

  def test_predict_with_both_limits(self):
    model = TestModel(dtype=tf.dtypes.float32)
    model.initialize()

    batch_size = 3
    max_instructions_in_batch = 10
    # Lengths of blocks in self.blocks_with_throughput are:
    # [1, 5, 1, 8, 3, 4, 1, 2, 9, 4].
    expected_batch_sizes = [3, 1, 3, 1, 1, 1]
    self.check_predict(
        model, batch_size, max_instructions_in_batch, expected_batch_sizes
    )

  def check_predict_deltas(self, model):
    with self.session() as sess:
      output_blocks = tuple(model.predict(sess, self.blocks))
      self.assertLen(output_blocks, len(self.blocks))

      for index, (in_block, out_block) in enumerate(
          zip(self.blocks, output_blocks)
      ):
        # Sum up delta predictions for the model with deltas.
        # predictions.
        num_instructions = len(in_block.instructions)

        # Inverse throughput on one prefix.
        expected_throughputs = []
        for task_index in range(model.num_tasks):
          pref_inv_throughputs = (index + 1 + task_index,)

          expected_throughputs.append(
              throughput.BasicBlockThroughput(
                  inverse_throughput_cycles=(
                      num_instructions * (index + 1 + task_index),
                  ),
                  prefix_inverse_throughput_cycles=(pref_inv_throughputs,)
                  * num_instructions,
              )
          )

        self.assertEqual(in_block, out_block.block)
        self.assertLen(out_block.throughputs, model.num_tasks)
        self.assertEqual(out_block.throughputs, expected_throughputs)

  def test_predict_deltas(self):
    model = TestModel(dtype=tf.dtypes.float32, use_deltas=True)
    model.initialize()

    self.check_predict_deltas(model)

  def test_predict_multi_task(self):
    task_list = ('task_1', 'task_2', 'task_3')
    model = TestModel(dtype=tf.dtypes.float32, task_list=task_list)
    model.initialize()

    self.check_predict(model, None, None, [len(self.blocks_with_throughput)])

  def test_predict_deltas_multi_task(self):
    task_list = ('task_1', 'task_2', 'task_3')
    model = TestModel(
        dtype=tf.dtypes.float32, task_list=task_list, use_deltas=True
    )
    model.initialize()

    self.check_predict_deltas(model)

  def test_schedule_batch_with_throughputs_multi_task(self):
    tasks = ['llvm', 'test']
    num_tasks = len(tasks)

    model = TestModel(dtype=tf.dtypes.float32, task_list=tasks)
    model.initialize()

    # Schedule a batch with no limits.
    full_schedule = model.schedule_batch(self.blocks_with_throughput)
    self.assertLen(self.blocks_with_throughput, model.num_visited_blocks)
    expected_outputs = full_schedule[model._expected_outputs]
    self.assertEqual(
        expected_outputs.shape, (len(self.blocks_with_throughput), num_tasks)
    )

    batch_size = 3
    model.num_visited_blocks = 0
    block_batch_schedule = model.schedule_batch(
        self.blocks_with_throughput, max_blocks_in_batch=batch_size
    )
    self.assertEqual(model.num_visited_blocks, batch_size)
    expected_outputs = block_batch_schedule[model._expected_outputs]
    self.assertEqual(expected_outputs.shape, (batch_size, num_tasks))

    with self.session() as sess:
      output = sess.run(model.output_tensor, feed_dict=full_schedule)
      self.assertAllEqual(
          output,
          [[x + 1, x + 2] for x in range(len(self.blocks_with_throughput))],
      )

      output = sess.run(model.output_tensor, feed_dict=block_batch_schedule)
      self.assertAllEqual(output, [[x + 1, x + 2] for x in range(batch_size)])

  def test_schedule_batch_with_throughputs_multi_task_with_deltas(self):
    tasks = ['llvm', 'test']
    num_tasks = len(tasks)

    model = TestModel(dtype=tf.dtypes.float32, task_list=tasks, use_deltas=True)
    model.initialize()

    # Only the first block has prefix timings for multiple tasks.
    blocks = self.blocks_with_throughput[:1]

    # Schedule a batch with no limits.
    full_schedule = model.schedule_batch(blocks)
    self.assertLen(blocks, model.num_visited_blocks)
    expected_outputs = full_schedule[model._expected_outputs]

    self.assertEqual(expected_outputs.shape, (len(blocks), num_tasks))
    expected_outputs_prefixes = full_schedule[model._expected_outputs_deltas]
    expected_num_prefixes = sum(
        len(block.block.instructions) for block in blocks
    )
    self.assertEqual(
        expected_outputs_prefixes.shape, (expected_num_prefixes, num_tasks)
    )

    with self.session() as sess:
      output_blocks, output_deltas = sess.run(
          (model.output_tensor, model.output_tensor_deltas),
          feed_dict=full_schedule,
      )

      expected_output_blocks = []
      expected_output_deltas = []
      for i, block in enumerate(blocks):
        block_expected_output = []
        for task in range(num_tasks):
          block_expected_output.append(
              (i + 1 + task) * len(block.block.instructions)
          )
        expected_output_blocks.append(block_expected_output)
        for _ in block.block.instructions:
          expected_output_deltas.append(
              [i + 1 + task for task in range(num_tasks)]
          )

      self.assertAllEqual(output_blocks, expected_output_blocks)
      self.assertAllEqual(output_deltas, expected_output_deltas)

  def test_training_with_full_variable_list(self):
    task_list = ['foo', 'bar']
    model = TestModelWithVarGroups(
        dtype=tf.dtypes.float32,
        use_deltas=False,
        learning_rate=0.1,
        task_list=task_list,
        trained_variable_groups=(
            TestModelWithVarGroups.WEIGHTS,
            TestModelWithVarGroups.BIAS,
        ),
    )
    model.initialize()
    with self.session() as sess:
      self.check_training_model(
          model,
          num_epochs=40,
          blocks=self.blocks_with_throughput[0:2],
          session=sess,
      )
      biases = sess.run(model._variable_groups[TestModelWithVarGroups.BIAS])
      for bias in biases:
        self.assertNotAlmostEqual(float(bias), -0.5)
      weights = sess.run(model._variable_groups[TestModelWithVarGroups.WEIGHTS])
      for weight in weights:
        self.assertNotAlmostEqual(float(weight), 0.5)

  def test_training_bias_only(self):
    task_list = ['foo', 'bar']
    model = TestModelWithVarGroups(
        dtype=tf.dtypes.float32,
        use_deltas=False,
        learning_rate=0.1,
        task_list=task_list,
        trained_variable_groups=(TestModelWithVarGroups.BIAS,),
    )
    model.initialize()
    with self.session() as sess:
      self.check_training_model(
          model,
          num_epochs=40,
          blocks=self.blocks_with_throughput[0:1],
          session=sess,
      )
      biases = sess.run(model._variable_groups[TestModelWithVarGroups.BIAS])
      for bias in biases:
        self.assertNotAlmostEqual(float(bias), -0.5)
      weights = sess.run(model._variable_groups[TestModelWithVarGroups.WEIGHTS])
      for weight in weights:
        self.assertAlmostEqual(float(weight), 0.5)

  def test_grad_clipping(self):
    task_list = ['foo', 'bar']
    model = TestModelWithVarGroups(
        dtype=tf.dtypes.float32,
        use_deltas=False,
        learning_rate=0.1,
        task_list=task_list,
        grad_clip_norm=1.0,
        trained_variable_groups=(TestModelWithVarGroups.BIAS,),
    )
    model.initialize()
    with self.session() as sess:
      self.check_training_model(
          model,
          num_epochs=40,
          blocks=self.blocks_with_throughput[0:1],
          session=sess,
      )

  def test_training_weight_only(self):
    task_list = ['foo', 'bar']
    model = TestModelWithVarGroups(
        dtype=tf.dtypes.float32,
        use_deltas=False,
        learning_rate=0.1,
        task_list=task_list,
        trained_variable_groups=(TestModelWithVarGroups.WEIGHTS,),
    )
    model.initialize()
    with self.session() as sess:
      self.check_training_model(
          model,
          num_epochs=40,
          blocks=self.blocks_with_throughput[0:1],
          session=sess,
      )
      biases = sess.run(model._variable_groups[TestModelWithVarGroups.BIAS])
      for bias in biases:
        self.assertAlmostEqual(float(bias), -0.5)
      weights = sess.run(model._variable_groups[TestModelWithVarGroups.WEIGHTS])
      for weight in weights:
        self.assertNotAlmostEqual(float(weight), 0.5)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
