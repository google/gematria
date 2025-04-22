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
"""Helper function for running inference with Gematria models."""

from collections.abc import Iterable
from typing import Optional

from absl import logging
from gematria.basic_block.python import throughput_protos
from gematria.model.python import model_base
from gematria.model.python import training
from gematria.proto import throughput_pb2


def _get_num_instructions_in_block_with_throughput_proto(
    proto: throughput_pb2.BasicBlockWithThroughputProto,
) -> int:
  return len(proto.basic_block.canonicalized_instructions)


def predict_for_protos(
    model: model_base.ModelBase,
    basic_blocks: Iterable[throughput_pb2.BasicBlockWithThroughputProto],
    max_blocks_in_batch: Optional[int] = None,
    max_instructions_in_batch: Optional[int] = None,
) -> Iterable[throughput_pb2.BasicBlockWithThroughputProto]:
  """Predicts the inverse throughput using the model.

  Assumes that model has been initialized and that it contains the appropriate
  weights. The input sequence is iterated through only once, and the method may
  safely be used with iterable objects that read the protos from a file or
  generate them on the fly.

  Args:
    model: The model used for inference.
    basic_blocks: The collection of basic blocks for which the inverse
      throughput is predicted.
    max_blocks_in_batch: The maximal number of basic blocks processed in a
      single batch. When not specified, the number of basic blocks in a batch is
      unlimited.
    max_instructions_in_batch: The maximal number of instructions across all
      basic blocks processed in a single batch. When not specified, the number
      of instructions in a batch is unlimited.

  Yields:
    The basic blocks from basic_blocks. Each basic block has a new
    inverse_throughputs value added to it with the prediction from the model.
  """
  batches = training.batches(
      basic_blocks,
      get_num_instructions=(
          _get_num_instructions_in_block_with_throughput_proto
      ),
      max_blocks_in_batch=max_blocks_in_batch,
      max_instructions_in_batch=max_instructions_in_batch,
  )
  for batch_index, protos in enumerate(batches):
    logging.info(
        'Processing proto batch %d (%d blocks).', batch_index, len(protos)
    )
    blocks = []
    block_is_valid = [False] * len(protos)
    for proto_index, proto in enumerate(protos):
      block = throughput_protos.block_with_throughput_from_proto(proto).block
      if model.validate_basic_block(block):
        block_is_valid[proto_index] = True
        blocks.append(block)

    # Blocks are already divided into batches according to the given criteria,
    # no need to use max_blocks_in_batch and max_instructions_in_batch again.
    predictions = iter(model.predict(blocks))

    # Inject predictions into the input protos.
    for proto, is_valid in zip(protos, block_is_valid):
      if is_valid:
        prediction = next(predictions)
        for task_index, task_predictions in zip(
            range(model.num_tasks), prediction.throughputs
        ):
          task_prefix_predictions = (
              task_predictions.prefix_inverse_throughput_cycles
          )
          task_throughput = proto.inverse_throughputs.add(
              source=model.get_source_name(task_index),
              inverse_throughput_cycles=(
                  task_predictions.inverse_throughput_cycles
              ),
          )
          for prefix_predictions in task_prefix_predictions:
            task_throughput.prefix_inverse_throughputs.add(
                inverse_throughput_cycles=prefix_predictions
            )
      yield proto
