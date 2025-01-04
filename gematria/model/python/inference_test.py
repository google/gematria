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

import copy

from gematria.model.python import inference
from gematria.model.python import model_base
from gematria.proto import throughput_pb2
from gematria.testing.python import model_test
import numpy as np
import tensorflow as tf

_PrefixThroughputProto = (
    throughput_pb2.ThroughputWithSourceProto.PrefixThroughputProto
)


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

  def _forward(self, feed_dict):
    if not self._use_deltas:
      return {'output': feed_dict['output']}
    else:
      return {'output_deltas': feed_dict['output_deltas']}

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
    output_name = 'output_deltas' if self._use_deltas else 'output'
    return {
        output_name: np.array(
            self._batch_collected_outputs, dtype=self.numpy_dtype
        ).reshape((-1, self.num_tasks)),
    }

  # @Override
  def schedule_batch(self, basic_blocks, *args, **kwargs):
    self.batch_sizes.append(len(basic_blocks))
    return super().schedule_batch(basic_blocks, *args, **kwargs)


class PredictForProtosTest(model_test.TestCase):
  num_blocks = 10

  def _check_predict(
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
      model: The model under test.
      max_blocks_in_batch: The maximal number of basic blocks in a batch, passed
        to model.predict().
      max_instructions_in_batch: The maximal number of instructions in a batch,
        passed to model.predict().
      expected_batch_sizes: A collection of expected sizes of batches processed
        by model.predict(), verified by this method.
    """
    # inference.predict_for_protos() modifies the protos in-place. We need to
    # make a copy to be able to compare them with the original protos.
    input_protos = copy.deepcopy(self.block_protos)
    output_protos = tuple(
        inference.predict_for_protos(
            model,
            input_protos,
            max_blocks_in_batch=max_blocks_in_batch,
            max_instructions_in_batch=max_instructions_in_batch,
        )
    )
    self.assertSequenceEqual(model.batch_sizes, expected_batch_sizes)
    self.assertLen(output_protos, len(self.block_protos))
    for index, (in_proto, out_proto) in enumerate(
        zip(self.block_protos, output_protos)
    ):
      # The prediction of the model is the number of calls to
      # model._add_basic_block_to_batch(). There is one call per basic block,
      # so we can get the expected value from the index of the basic block.
      expected_inverse_throughputs = [*in_proto.inverse_throughputs]
      for task_index in range(model.num_tasks):
        expected_inverse_throughputs.append(
            throughput_pb2.ThroughputWithSourceProto(
                source=model.get_source_name(task_index),
                inverse_throughput_cycles=(index + 1 + task_index,),
            )
        )
      self.assertEqual(in_proto.basic_block, out_proto.basic_block)
      # NOTE(ondrasej): assertSequenceEqual refuses to compare a repeated
      # field of a proto with a native sequence type.
      self.assertSequenceEqual(
          tuple(out_proto.inverse_throughputs), expected_inverse_throughputs
      )

  def test_predict_single_batch(self):
    model = TestModel(dtype=tf.dtypes.float32)
    model.initialize()
    self._check_predict(model, None, None, (len(self.block_protos),))

  def test_predict_multiple_batches(self):
    model = TestModel(dtype=tf.dtypes.float32)
    model.initialize()

    batch_size = 3
    num_blocks = len(self.blocks_with_throughput)
    expected_batch_sizes = [batch_size] * (num_blocks // batch_size) + [
        num_blocks % batch_size
    ]
    self._check_predict(model, 3, None, expected_batch_sizes)

  def test_predict_with_instruction_limit(self):
    model = TestModel(dtype=tf.dtypes.float32)
    model.initialize()

    max_instructions_in_batch = 10
    # Lengths of blocks in self.blocks_with_throughput are:
    # [1, 5, 1, 8, 3, 4, 1, 2, 9, 4].
    expected_batch_sizes = [3, 1, 4, 1, 1]
    self._check_predict(
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
    self._check_predict(
        model, batch_size, max_instructions_in_batch, expected_batch_sizes
    )

  def test_predict_multi_task(self):
    task_list = ('task_1', 'task_2', 'task_3')
    model = TestModel(dtype=tf.dtypes.float32, task_list=task_list)
    model.initialize()

    self._check_predict(model, None, None, [len(self.blocks_with_throughput)])

  def check_predict_deltas(self, model):
    """Checks the prediction of the model when predicting also deltas."""
    input_protos = copy.deepcopy(self.block_protos)
    output_protos = tuple(
        inference.predict_for_protos(model, input_protos)
    )
    self.assertLen(output_protos, len(self.block_protos))

    for index, (in_proto, out_proto) in enumerate(
        zip(self.block_protos, output_protos)
    ):
      # Sum up delta predictions for the model with deltas.
      # predictions.
      num_instructions = len(in_proto.basic_block.canonicalized_instructions)

      # Inverse throughput on one prefix.
      expected_throughputs = [*in_proto.inverse_throughputs]
      for task_index in range(model.num_tasks):
        pref_inv_throughputs = _PrefixThroughputProto(
            inverse_throughput_cycles=(index + 1 + task_index,)
        )

        expected_throughputs.append(
            throughput_pb2.ThroughputWithSourceProto(
                source=model.get_source_name(task_index),
                inverse_throughput_cycles=(
                    num_instructions * (index + 1 + task_index),
                ),
                prefix_inverse_throughputs=(
                    num_instructions * (pref_inv_throughputs,)
                ),
            )
        )

      self.assertEqual(in_proto.basic_block, out_proto.basic_block)
      self.assertSequenceEqual(
          tuple(out_proto.inverse_throughputs), expected_throughputs
      )

  def test_predict_deltas(self):
    model = TestModel(dtype=tf.dtypes.float32, use_deltas=True)
    model.initialize()

    self.check_predict_deltas(model)

  def test_predict_deltas_multi_task(self):
    task_list = ('task_1', 'task_2', 'task_3')
    model = TestModel(
        dtype=tf.dtypes.float32, task_list=task_list, use_deltas=True
    )
    model.initialize()

    self.check_predict_deltas(model)


if __name__ == '__main__':
  tf.test.main()
