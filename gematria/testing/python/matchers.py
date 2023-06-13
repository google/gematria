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
"""Matchers that can be used with unittest.mock."""

from collections.abc import Sequence
from typing import Any


class SequenceEqual:
  """Matches sequence-like objects elementwise.

  Checks only the elements of the sequences, but does not check the type of the
  sequence object.
  """

  def __init__(self, sequence: Sequence[Any]):
    """Initializes the matcher with a sequence to match against."""
    self._sequence = sequence

  def __eq__(self, other: Any) -> bool:
    return (isinstance(other, Sequence) and
            len(self._sequence) == len(other) and
            all(own_item == other_item
                for own_item, other_item in zip(self._sequence, other)))

  def __repr__(self) -> str:
    return f"SequenceEqual({self._sequence!r})"
