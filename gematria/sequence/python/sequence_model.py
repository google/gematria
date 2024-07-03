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
"""Base class for Gematria models that read basic blocks as sequences of tokens."""

import abc
from collections.abc import MutableSequence
from typing import Optional

from gematria.basic_block.python import basic_block
from gematria.model.python import model_base
from gematria.model.python import oov_token_behavior
from gematria.model.python import token_model
import numpy as np
import tensorflow.compat.v1 as tf
import tf_keras as keras

_OutOfVocabularyTokenBehavior = oov_token_behavior.OutOfVocabularyTokenBehavior


class SequenceModelBase(token_model.TokenModel, model_base.ModelBase):
  """Base class for models that treat basic blocks as sequences of tokens.

  Represents basic blocks as a hierarchy of sequences of tokens:
  - Each basic block is represented as a sequence of instructions,
  - Each instruction is represented as a sequence of tokens from a fixed
    vocabulary, usually based on the assembly code of the instruction.

  In the model, these sequences are represented as:
  - a single flat sequence of tokens. This sequence has tokens from instructions
    in all basic blocks in the natural order:
      T(0, 0, 0), T(0, 0, 1), ..., T(0, 0, num_tokens(0, 0)),
      T(0, 1, 0), T(0, 1, 1), ..., T(0, 1, num_tokens(0, 1)),
      ..
      T(0, num_instructions(0), 0), ...,
      T(1, 0, 0), ...,
      ...
    where T(b, i, t) is the t-th token of i-th instruction of b-th block in the
    batch, num_tokens(b, i) is the number of tokens of i-th instruction of b-th
    block, and num_instructions(b) is the number of instructions in b-th block.
  - a single tensor that contains the number of tokens in each instruction, in
    the natural order:
      num_tokens(0, 0), ..., num_tokens(0, num_instructions(0)),
      num_tokens(1, 0), ...
  - a single tensor that contains the number of instructions in each block, in
    the natural order:
      num_instructions(0), num_instructions(1), ...

  The tokens are represented as their indices in the vocabulary list passed to
  the constructor of the model. This class translates tokens from strings to
  their numeric representations, and assumes that the Keras model uses the same
  mapping from string tokens to indices.

  This class automatically handles out-of-vocabulary tokens:
    - either they can be replaced by a specified token from the vocabulary,
    - or they cause an error and abort the training.
  The desired behavior is configured by the `out_of_vocabulary_behavior`
  argument of the constructor. Basic blocks that contains out of vocabulary
  tokens can be detected using `model.validate_basic_block()`.

  The actual processing is done by a Keras model that takes the inputs in the
  format described above, and returns a tensor with output values.
  """

  # The model used for processing the data.
  _model: Optional[keras.Model] = None

  # Temporary lists used when transforming basic blocks from BasicBlockProto to
  # the three tensors described above.
  _batch_tokens: MutableSequence[int]
  _batch_num_tokens_per_instruction: MutableSequence[int]
  _batch_num_instructions_per_block: MutableSequence[int]

  # Inputs (tf.placeholder tensors) specific to sequence models.
  _token_sequence_placeholder: tf.Tensor
  _num_tokens_per_instruction_placeholder: tf.Tensor
  _num_instructions_per_block_placeholder: tf.Tensor

  @abc.abstractmethod
  def _create_model(self) -> keras.Model:
    """Creates the Keras model for this class.

    The returned model must take a tuple of 1D int32 tensors
    (token_sequence, num_tokens_per_instruction, num_instructions_per_block) as
    its input, and return a single tensor of the shape (output_size, num_tasks)
    as its output, where output_size is the number of blocks in the batch in the
    seq2num mode, and the number of instructions in the batch in the seq2seq
    mode.

    Returns:
      The Keras model matching the input/output specification.
    """

  def _create_tf_graph(self) -> None:
    super()._create_tf_graph()
    self._model = self._create_model()
    self._token_sequence_placeholder = tf.placeholder(
        dtype=tf.dtypes.int32,
        shape=(None,),
        name='SequenceModelBase.token_sequence',
    )
    self._num_tokens_per_instruction_placeholder = tf.placeholder(
        dtype=tf.dtypes.int32,
        shape=(None,),
        name='SequenceModelBase.num_tokens_per_instruction',
    )
    self._num_instructions_per_block_placeholder = tf.placeholder(
        dtype=tf.dtypes.int32,
        shape=(None,),
        name='SequenceModelBase.num_instructions_per_block',
    )

    model_output = self._model(
        (
            self._token_sequence_placeholder,
            self._num_tokens_per_instruction_placeholder,
            self._num_instructions_per_block_placeholder,
        )
    )

    if self._use_deltas:
      self._output_tensor_deltas = model_output
    else:
      self._output_tensor = model_output

  # @Override
  def _start_batch(self) -> None:
    """See base class."""
    super()._start_batch()
    self._batch_tokens = []
    self._batch_num_tokens_per_instruction = []
    self._batch_num_instructions_per_block = []

  # @Override
  def _make_batch_feed_dict(self) -> model_base.FeedDict:
    """See base class."""
    batch_tokens = np.array(self._batch_tokens, dtype=np.int32)
    if self._oov_injection_probability > 0:
      oov_injection_mask = (
          np.random.uniform(0.0, 1.0, size=batch_tokens.shape)
          < self._oov_injection_probability
      )
      batch_tokens[oov_injection_mask] = self._oov_token

    return {
        self._token_sequence_placeholder: batch_tokens,
        self._num_tokens_per_instruction_placeholder: np.array(
            self._batch_num_tokens_per_instruction, dtype=np.int32
        ),
        self._num_instructions_per_block_placeholder: np.array(
            self._batch_num_instructions_per_block, dtype=np.int32
        ),
    }

  # @Override
  def _add_basic_block_to_batch(self, block: basic_block.BasicBlock) -> None:
    """See base class."""
    self._batch_num_instructions_per_block.append(len(block.instructions))
    for instruction in block.instructions:
      tokens = instruction.as_token_list()
      self._batch_num_tokens_per_instruction.append(len(tokens))
      self._batch_tokens.extend(self._token_index(token) for token in tokens)
