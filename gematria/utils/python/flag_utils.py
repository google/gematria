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
"""Helper functions for defining and validating command-line flags."""

from collections.abc import Sequence

# Common error messages for flag validators.
MUST_BE_POSITIVE_INTEGER_LIST_ERROR = (
    'Flag must contain a list of positive integers.')
MUST_BE_POSITIVE_ERROR = 'Flag must contain a positive number.'
MUST_BE_BETWEEN_ZERO_AND_ONE = 'Flag must be a float between zero and one.'


def layers_from_str(layers: str) -> Sequence[int]:
  """Converts a comma-separated list of integers to an actual list of ints."""
  layer_sizes = filter(None, layers.strip().split(','))
  return tuple(int(size) for size in layer_sizes)


def is_between_zero_and_one(value: float) -> bool:
  """Returns true if the value is between zero and one."""
  return value >= 0.0 and value <= 1.0


def is_positive(value: int) -> bool:
  """Returns True if the value is greater than zero."""
  return value > 0


def is_positive_integer_list(values: str) -> bool:
  """Checks that all values in the input parse as positive integers."""
  # Check for the empty strings, for the cases where the layers are empty.
  if values.isspace() or not values:
    return True
  try:
    return all(int(value) > 0 for value in values.split(','))
  except ValueError:
    return False
