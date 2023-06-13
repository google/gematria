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
"""Command-line flags for models based on TokenModel.

This module defines command-line flag --gematria_tokens_file and a function for
extracting tokens from the file specified by this command-line flag.
"""

from collections.abc import Sequence
from typing import Optional

from absl import flags

import tensorflow.compat.v1 as tf

from gematria.model.python import oov_token_behavior
from gematria.utils.python import flag_utils

_TOKEN_FILE = flags.DEFINE_string(
    'gematria_tokens_file',
    None,
    ('The text file that contains the list of tokens used in the input'
     ' basic blocks. Used to create the token embedding table in models that'
     ' use a token representation of instructions. Assumes that the argument'
     ' is the path of a text file that contains one token per line. Lines'
     ' that start with a hash symbol (#) are considered as comments and'
     ' ignored.'),
)

_OOV_REPLACEMENT_TOKEN = flags.DEFINE_string(
    'gematria_out_of_vocabulary_replacement_token',
    None,
    ('The token used as a replacement when the input contains an'
     ' out-of-vocabulary token. When empty, no replacement is done and the'
     ' model returns an error'),
)

OUT_OF_VOCABULARY_INJECTION_PROBABILITY = flags.DEFINE_float(
    'gematria_out_of_vocabulary_injection_probability',
    0.0,
    ('The probability of replacing a token in the graph with the token'
     ' specified --gematria_out_of_vocabulary_replacement_token. This flag'
     ' may be used only when --gematria_out_of_vocabulary_replacement_token'
     ' is non-empty.'),
)

flags.register_validator(
    OUT_OF_VOCABULARY_INJECTION_PROBABILITY.name,
    flag_utils.is_between_zero_and_one,
    flag_utils.MUST_BE_BETWEEN_ZERO_AND_ONE,
)


@flags.multi_flags_validator(
    (_OOV_REPLACEMENT_TOKEN.name, OUT_OF_VOCABULARY_INJECTION_PROBABILITY.name),
    message=('Replacement token must be provided when'
             ' --gematria_out_of_vocabulary_injection_probability is not 0.0.'),
)
def _out_of_vocabulary_injection_is_valid(flags_dict):
  """Checks that the out-of-vocabulary injection flags are valid."""
  replacement_token = flags_dict[_OOV_REPLACEMENT_TOKEN.name]
  injection_probability = flags_dict[
      OUT_OF_VOCABULARY_INJECTION_PROBABILITY.name]
  return injection_probability == 0.0 or bool(replacement_token)


def get_oov_token_behavior_from_command_line_flags(
) -> (oov_token_behavior.OutOfVocabularyTokenBehavior):
  """Returns the out-of-vocabulary token behavior from command-line flags."""
  if not _OOV_REPLACEMENT_TOKEN.value:
    return oov_token_behavior.OutOfVocabularyTokenBehavior.return_error()
  return oov_token_behavior.OutOfVocabularyTokenBehavior.replace_with_token(
      _OOV_REPLACEMENT_TOKEN.value)


def get_tokens_from_command_line_flags(  #
    model_tokens: Sequence[str] = (),) -> Optional[Sequence[str]]:
  """Returns the list of tokens used in the model.

  When the command-line flag --gematria_tokens_file is used, returns a sorted
  list of tokens from this file merged with `model_tokens`. The input file is
  expected to be a text file that contains one token per line. When loading the
  tokens, the function:
    1. removes leading and trailing whitespace from each line.
    2. ignores lines starting with a hash character (#).

  Args:
    model_tokens: Extra tokens that may be used by the internal representation
      of basic blocks in the model, but that do not appear in the instructions.

  Returns:
    The list of tokens loaded from the file specified by the command-line flags
    or None when no such file is specified. The returned list is sorted and
    contains each token at most once.
  """
  if not _TOKEN_FILE.value:
    return None

  tokens = set(model_tokens)
  with tf.io.gfile.GFile(_TOKEN_FILE.value, 'r') as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith('#'):
        continue
      tokens.add(line)

  return sorted(tokens)


def mark_token_flags_as_required() -> None:
  """Adds flag validators to make sure that the token list flags are provided."""
  flags.mark_flag_as_required(_TOKEN_FILE.name)


def set_default_oov_replacement_token(token: Optional[str]) -> None:
  """Overrides the default out-of-vocabulary replacement token."""
  flags.set_default(_OOV_REPLACEMENT_TOKEN, token)
