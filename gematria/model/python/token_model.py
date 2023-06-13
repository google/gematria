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
"""Common functionality for models that represent instructions using asm tokens.

Provides TokenModel, a class that can be used as a mixin with other Gematria
model classes and provides methods and attributes for working with tokens.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Optional

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from gematria.basic_block.python import basic_block
from gematria.model.python import model_base
from gematria.model.python import oov_token_behavior

_OutOfVocabularyTokenBehavior = oov_token_behavior.OutOfVocabularyTokenBehavior


class TokenNotFoundError(Exception):
  """The exception raised when the model encounters an unknown token.

  Attributes:
    token: The name of the token that was not found.
  """

  def __init__(self, token: str):
    super().__init__(f'Token not found: "{token}"')
    self.token = token


class TokenModel(model_base.ModelBase):
  """A mixin class that adds support for tokens to a Gematria model.

  This class is intended to be used either as a standalone class that models
  derive from, or as a mixin class used together with another class that derives
  from model_base.ModelBase.
  """

  # NOTE(ondrasej): The method resolution order in Python ensures that the
  # initialization of each class is done only once, and the order respects the
  # inheritance order. See https://www.python.org/download/releases/2.3/mro/ for
  # more details.

  # The name of TokenModel.token_list_tensor in the TensorFlow graph.
  TOKENS_TENSOR_NAME = 'TokenModel.token_list'

  # The following tensors encode the parameters of the graph builder used with
  # the model, so that the graph builder can be reconstructed from a trained and
  # exported model.

  # A 1D byte tensor that contains the list of tokens in the order of their
  # indices in the graph builder. See the docstring of self.token_list_tensor
  # for more details on the format of the data.
  _token_list_tensor: tf.Tensor

  # The list of token names, in the order of their indices in the model.
  _token_list: Sequence[str]

  # Mapping from token names to their indices.
  _tokens: Mapping[str, int]

  # The out-of-vocabulary token behavior provided during the construction of
  # this class.
  _oov_behavior: oov_token_behavior.OutOfVocabularyTokenBehavior

  # The index of the out-of-vocabulary replacement token in self._token_list.
  # This is set only when the model is initialized with the unknown token
  # replacement strategy; otherwise, it is None.
  _oov_token: Optional[int]

  # The probability that the model will inject an out-of-vocabulary token into
  # the token stream during training.
  # NOTE(ondrasej): The actual injecting depends on the data representation in
  # the model and it needs to be implemented for each class of models
  # separately.
  _oov_injection_probability: float

  def __init__(
      self,
      *,
      tokens: Iterable[str],
      out_of_vocabulary_behavior: _OutOfVocabularyTokenBehavior,
      out_of_vocabulary_injection_probability: float = 0.0,
      **kwargs,
  ):
    """Initializes the token model mixin.

    Args:
      tokens: The list of all tokens that can appear in the basic blocks.
      out_of_vocabulary_behavior: Specification of the behavior of the model
        when it encounters an unknown token.
      out_of_vocabulary_injection_probability: The probability that the model
        will inject an out-of-vocabulary token into the token stream during
        training.
      **kwargs: All other arguments are passed to the next constructor in the
        chain.

    Raises:
      ValueError: When the out of vocabulary replacement token is not in tokens.
    """
    replace_token = _OutOfVocabularyTokenBehavior.BehaviorType.REPLACE_TOKEN
    if (out_of_vocabulary_injection_probability > 0.0 and
        out_of_vocabulary_behavior.behavior_type != replace_token):
      raise ValueError(
          'out_of_vocabulary_injection_probability may be non-zero only when'
          ' out_of_vocabulary_behavior.behavior_type is kReplaceToken.')
    self._oov_injection_probability = out_of_vocabulary_injection_probability

    self._token_list = tuple(tokens)
    self._tokens = {token: i for i, token in enumerate(self._token_list)}
    self._oov_behavior = out_of_vocabulary_behavior
    self._oov_token = None
    if (out_of_vocabulary_behavior.behavior_type ==
        _OutOfVocabularyTokenBehavior.BehaviorType.REPLACE_TOKEN):
      replacement_token = out_of_vocabulary_behavior.replacement_token
      self._oov_token = self._tokens.get(replacement_token)
      if self._oov_token is None:
        raise ValueError(f'Token {replacement_token} was not found in tokens.')

    super().__init__(**kwargs)

  @property
  def token_list_tensor(self) -> tf.Tensor:
    """Returns the tensor that contains the list of node tokens.

    The tensor is a byte (uint8) tensor that contains the token names encoded
    using the utf-8 encoding; the tokens are in the order of their indices in
    the basic block graph builder object, and they are separated by a zero byte.
    """
    return self._token_list_tensor

  @property
  def output_tensor_names(self) -> Sequence[str]:
    """See base class."""
    return (
        *super().output_tensor_names,
        TokenModel.TOKENS_TENSOR_NAME,
    )

  def _token_index(self, token: str) -> int:
    """Returns token index for the given token.

    Args:
      token: The name of the token.

    Returns:
      The index of the token, or the index of the out-of-vocabulary token when
      the token is not in the vocabulary, and the out-of-vocabulary behavior is
      to return a replacement token.

    Raises:
      TokenNotFoundError: When the token is not in the vocabulary, and the
        out-of-vocabulary token behavior is to return an error.
    """
    token_index = self._tokens.get(token)
    if token_index is None:
      if self._oov_token is not None:
        token_index = self._oov_token
      else:
        raise TokenNotFoundError(token)
    return token_index

  def validate_basic_block(self, block: basic_block.BasicBlock) -> bool:
    """See base class."""
    return self.validate_basic_blockTokens(block)

  def validate_basic_blockTokens(self, block: basic_block.BasicBlock) -> bool:
    """Checks that basic_block_proto uses only known tokens."""
    if self._oov_token is not None:
      # All blocks are valid when the policy is to replace unknown tokens.
      return True
    for instruction in block.instructions:
      for token in instruction.as_token_list():
        if token not in self._tokens:
          logging.error('Unknown token: "%s"', token)
          return False
    return True

  def _create_tf_graph(self):
    """See base class."""
    super()._create_tf_graph()
    # Convert the token list into an array of bytes. We need to go through NumPy
    # because tf.constant() always treats a bytes() object as a string and can't
    # use it with any other dtype.
    token_list_array = np.frombuffer(
        b'\0'.join(token.encode('utf-8') for token in self._token_list),
        dtype=np.uint8,
    )
    self._token_list_tensor = tf.constant(
        token_list_array, name=TokenModel.TOKENS_TENSOR_NAME)
