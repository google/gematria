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
"""Contains an implementation of the Hierarchical LSTM model.

See the docstring of HierarchicalLstmModel for more details of the model.
"""

from collections.abc import Sequence

import tensorflow.compat.v1 as tf

from gematria.model.python import model_base
from gematria.sequence.python import sequence_model


class HierarchicalLstmModel(sequence_model.SequenceModelBase):
  """The implementation of a Hierarchical LSTM model [1].

  This model is a generalization of the hierarchical LSTM model described in the
  Ithemal paper [1]. The model uses a representation of basic blocks as
  sequences of sequences of tokens. The tokens are based on the assembly code of
  the instructions; each "inner" sequence contains all tokens belonging to one
  instruction; each "outer" sequence contains all instructions of one basic
  block

  In addition to the model described in the Ithemal paper, this model also
  includes modifications described in the GRANITE paper [2]:
    * a sequence to sequence mode. The model can make either one prediction per
      basic block (seq2num; this is the model from the Ithemal paper) or one
      prediction per instruction in a basic block (seq2seq). In both modes, the
      computation of instruction embeddings and the structure of the basic-block
      level of the network remain the same, the difference is only in the way
      the output value is computed.
    * a more complex decoder network. Instead of using a single dot product
      operation to compute the predictions from basic block embeddings computed
      by the LSTM network, this model allows using a decoder that contains a
      given number of dense ReLU layers and a dot product at the end. The number
      of layers and their sizes are hyper-parameters of the model.
      Using zero dense layers is equivalent to the model from the Ithemal paper.
    * a way to train the model in multi-task mode, i.e. train the model for
      multiple microarchitectures at the same time. In this mode, the LSTM
      networks are shared between the microarchitectures. The decoder can be
      split into a shared part, and independent task-specific parts.
    * an option to use a bidirectional LSTM network at the basic block level.

  The model has three levels:
  1. Token level: Each token extracted from the assembly code is represented as
     an embedding vector of size self.token_embedding_size. The token embedding
     vectors are learned parameters of the model.
  2. Instruction level: For each instruction, the model computes an instruction
     embedding vector of size self._instruction_embedding_size from the
     embedding vectors of the tokens of the instruction. The token embedding
     vectors are processed by an LSTM cell chain; a concatenation of the
     state+output vectors of the last cell in the chain is used as the
     instruction embedding vector.
  3. Basic block level: The model computes a basic block embedding vector by
     presenting all instruction embedding vectors from the block to another LSTM
     chain.
  4.1. In the seq2num mode, the inverse throughput prediction is computed by
       presenting the concatenated state+output vectors of the last LSTM cell
       from step 3. to the decoder network.
  4.2. In the seq2seq mode, the delta for each instruction is computed by
       presenting the output vector of each LSTM cell from step 3. to the
       decoder network.
       Note that due to API limitations in the Keras LSTM implementation, we can
       access only the state vector of the last LSTM cell in an LSTM chain, for
       other cells, we get only their output vectors. Hence, the inputs passed
       to the decoder network have only half the size of the vector used in
       seq2num mode.

  [1] Ithemal: Accurate, Portable and Fast Basic Block Throughput Estimation
      using Deep Neural Networks, https://arxiv.org/abs/1808.07412.
  [2] GRANITE: A Graph Neural Network Model for Basic Block Throughput
      Estimation, https://arxiv.org/abs/2210.03894.
  """

  # The size of token embedding vectors used in the model.
  _token_embedding_size: int

  # The size of the instruction embedding vectors used in the model.
  _instruction_embedding_size: int

  # The size of the basic block embedding vectors used in the model.
  _block_embedding_size: int

  # When True, the model uses a bidirectional LSTM network at the basic block
  # level.
  _bidirectional: bool

  # The list of dense layer sizes in the shared part of the decoder.
  _output_layers: Sequence[int]

  # The list of dense layer sizes in the task-specific part of the decoder.
  _task_output_layers: Sequence[int]

  def __init__(
      self,
      *,
      token_embedding_size: int,
      instruction_embedding_size: int,
      block_embedding_size: int,
      output_layers: Sequence[int],
      task_output_layers: Sequence[int],
      bidirectional: bool,
      **kwargs,
  ):
    """Initializes the hierarchical LSTM model.

    Args:
      token_embedding_size: The size of token embedding vectors used in the
        model.
      instruction_embedding_size: The size of the instruction embedding vectors
        used in the model. These are the embedding vectors computed by the
        instruction-level LSTM network.
      block_embedding_size: The size of the basic block embedding vectors used
        in the model. These are the embedding vectors computed by the basic
        block level LSTM networks. Note that in the seq2seq mode, only one half
        of this size is passed to the decoder, due to a limitation of the Keras
        LSTM API.
      output_layers: The list of dense layer sizes in the shared part of the
        decoder. The layer at the first index is the input layer of the decoder.
        When empty, no shared dense layers are used in the decoder.
      task_output_layers: The list of dense layer sizes in the task-specific
        part of the decoder. The layer at the first index is the input layer of
        the task-specific part of the decoder. When empty, no task-specific
        dense layers are used in the decoder, only the (task-specific) dot
        product at the end.
      bidirectional: When True, the model uses a bidirectional LSTM network at
        the basic block level.
      **kwargs: All remaining keyword args are passed to the constructor of the
        base class.
    """
    super().__init__(**kwargs)
    self._token_embedding_size = token_embedding_size
    self._instruction_embedding_size = instruction_embedding_size
    self._block_embedding_size = block_embedding_size
    self._bidirectional = bidirectional
    self._output_layers = tuple(output_layers)
    self._task_output_layers = tuple(task_output_layers)

  # @Override
  def _create_model(self) -> tf.keras.Model:
    """See base class."""
    return _HierarchicalLstmKerasModel(
        num_tokens=len(self._token_list),
        task_list=self.task_list,
        token_embedding_size=self._token_embedding_size,
        instruction_embedding_size=self._instruction_embedding_size,
        block_embedding_size=self._block_embedding_size,
        bidirectional=self._bidirectional,
        output_layers=self._output_layers,
        task_output_layers=self._task_output_layers,
        use_deltas=self.use_deltas,
        dtype=self.dtype,
    )

  # @Override
  def _make_model_name(self):
    """See base class."""
    return ('HierarchicalLSTM:'
            f' use_deltas={self._use_deltas}'
            f' token_embedding_size={self._token_embedding_size},'
            f' instruction_embedding_size={self._instruction_embedding_size},'
            f' block_embedding_size={self._block_embedding_size},'
            f' bidirectional={self._bidirectional},'
            f' output_layers={self._output_layers!r},'
            f' task_output_layers={self._task_output_layers!r}')


class _HierarchicalLstmKerasModel(tf.keras.Model):
  """Implements the hierarchical LSTM model as a Keras model.

  The model works both in the sequence-to-number mode (the model returns one
  prediction per basic block) and in a sequence-to-sequence mode (the model
  returns one prediction for each instruction; these predictions are summed to
  get the basic block prediction).

  The inputs and outputs of the model conform to the format used by
  sequence_model.SequenceModelBase.
  """

  def __init__(
      self,
      num_tokens: int,
      task_list: Sequence[str],
      token_embedding_size: int,
      instruction_embedding_size: int,
      block_embedding_size: int,
      bidirectional: bool,
      use_deltas: bool,
      output_layers: Sequence[int],
      task_output_layers: Sequence[int],
      dtype: tf.dtypes.DType,
      name: str = 'HierarchicalLstmModel',
      **kwargs,
  ):
    """Initializes the hierarchical LSTM model.

    Args:
      num_tokens: The number of tokens in the vocabulary.
      task_list: The names of the tasks performed by the model.
      token_embedding_size: The size of learned token embedding vectors.
      instruction_embedding_size: The size of instruction embedding vectors;
        these vectors are the output of the LSTM network processing token
        embedding vectors (the inner LSTM).
      block_embedding_size: The size of basic block embedding vectors; these
        vectors are the output of the LSTM network processing instruction
        embedding vectors (the outer LSTM).
      bidirectional: When True, the model uses a bidirectional LSTM network at
        the basic block level.
      use_deltas: True to create a sequence-to-sequence model; False for a
        sequence-to-number model.
      output_layers: The number of layers of a dense network that is shared
        between tasks and that is used for post-processing of the outputs of the
        RNN.
      task_output_layers: The number of layers of a dense network that is
        task-specific and that is used for post-processing of the outputs of the
        RNN and the output head network.
      dtype: The DType used for tensors in the model.
      name: The name of the model.
      **kwargs: All remaining (keyword) arguments are passed to tf.keras.Model
        constructor.
    """
    if num_tokens <= 0:
      raise ValueError('num_tokens must be positive')
    if not task_list:
      raise ValueError('task list must not be empty')
    if token_embedding_size <= 0:
      raise ValueError('token_embedding size must be positive')
    if instruction_embedding_size <= 0 or instruction_embedding_size % 2 == 1:
      raise ValueError(
          'instruction_embedding_size must be a positive even number, is'
          f' {instruction_embedding_size}')
    if block_embedding_size <= 0 or block_embedding_size % 2 == 1:
      raise ValueError('block_embedding_size must be a positive even number, is'
                       f' {block_embedding_size}')

    super().__init__(name=name, **kwargs)

    self._use_deltas = use_deltas
    self._task_list = task_list

    self._num_tokens = num_tokens
    self._token_embedding_size = token_embedding_size
    self._instruction_embedding_size = instruction_embedding_size
    self._block_embedding_size = block_embedding_size
    self._bidirectional = bidirectional

    self._ragged_from_row_lengths = _RaggedFromRowLengths()
    self._values_from_ragged = _RaggedTensorValues()
    self._concatenate_outputs = tf.keras.layers.Concatenate(axis=1)

    self._token_embedding = tf.keras.layers.Embedding(
        input_dim=self._num_tokens,
        output_dim=self._token_embedding_size,
        dtype=dtype,
    )

    self._instruction_lstm = tf.keras.layers.LSTM(
        units=instruction_embedding_size // 2, return_state=True, dtype=dtype)

    self._block_lstm = tf.keras.layers.LSTM(
        units=block_embedding_size // 2,
        return_sequences=self._use_deltas,
        return_state=not self._use_deltas,
        dtype=dtype,
    )
    if self._bidirectional:
      self._block_lstm = tf.keras.layers.Bidirectional(self._block_lstm)

    self._output_network = tf.keras.Sequential()
    for layer_size in output_layers:
      self._output_network.add(
          tf.keras.layers.Dense(
              layer_size,
              activation=tf.keras.activations.relu,
              dtype=dtype,
              bias_initializer='glorot_normal',
          ))

    # Create each output head as an independent layers, so that training for one
    # task does not affect the others.
    self._output_heads = []
    for task_name in task_list:
      head = tf.keras.Sequential()
      for layer_size in task_output_layers:
        head.add(
            tf.keras.layers.Dense(
                layer_size,
                activation=tf.keras.activations.relu,
                dtype=dtype,
                bias_initializer='glorot_normal',
            ))
      head.add(
          tf.keras.layers.Dense(
              1,
              activation='linear',
              use_bias=False,
              dtype=dtype,
              name=f'output_{task_name}',
          ))
      self._output_heads.append(head)

  def call(
      self,
      inputs: tuple[
          tf.types.experimental.TensorLike,
          tf.types.experimental.TensorLike,
          tf.types.experimental.TensorLike,
      ],
  ) -> tf.types.experimental.TensorLike:
    """Applies the model to basic block inputs.

    Args:
      inputs: The basic block data in the format specified in the documentation
        of SequenceModelBase.

    Returns:
      The output of the model. Depending on the value of `use_deltas` during the
      initialization, this contains either predictions for each basic block in
      the input batch (when use_deltas is False), or for each instruction in the
      batch (when use_deltas is True).
    """
    (
        token_sequence,
        num_tokens_per_instruction,
        num_instructions_per_block,
    ) = inputs

    embedded_tokens = self._token_embedding(token_sequence)
    instructions = self._ragged_from_row_lengths(embedded_tokens,
                                                 num_tokens_per_instruction)
    instruction_lstm_outputs = self._instruction_lstm(instructions)
    embedded_instructions = self._concatenate_outputs(
        (instruction_lstm_outputs[1], instruction_lstm_outputs[2]))

    blocks = self._ragged_from_row_lengths(embedded_instructions,
                                           num_instructions_per_block)
    block_lstm_outputs = self._block_lstm(blocks)

    if self._use_deltas:
      # In seq2seq, block_lstm_outputs is a tensor that contains the outputs of
      # each basic block level LSTM cell.
      output_embedding = self._values_from_ragged(block_lstm_outputs)
    else:
      # In seq2num, block_lstm_outputs is a tuple that contains
      #   1. the outputs of the last basic-block level LSTM cell.
      #   2. the hidden states of the last basic-block level LSTM cell.
      #   3. the outputs of the last basic-block level LSTM cell (again).
      output_embedding = self._concatenate_outputs(
          (block_lstm_outputs[1], block_lstm_outputs[2]))

    outputs = tuple(
        output_head(output_embedding) for output_head in self._output_heads)
    output_name = (
        model_base.ModelBase.OUTPUT_TENSOR_DELTAS_NAME
        if self._use_deltas else model_base.ModelBase.OUTPUT_TENSOR_NAME)
    return tf.keras.layers.concatenate(outputs, axis=1, name=output_name)


class _RaggedFromRowLengths(tf.keras.layers.Layer):
  """Creates a ragged tensor from values and row lengths."""

  def call(self, values, row_lengths):
    """Applies the layer to a values tensor and a row lengths tensor."""
    return tf.RaggedTensor.from_row_lengths(
        values=values, row_lengths=row_lengths)


class _RaggedTensorValues(tf.keras.layers.Layer):
  """Extracts values from a ragged tensor."""

  def call(self, ragged_tensor):
    """Applies the layer to a ragged tensor."""
    return ragged_tensor.values
