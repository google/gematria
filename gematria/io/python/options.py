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
"""Parameters for loading and saving data."""

import enum


@enum.unique
class ThroughputSelection(enum.Enum):
  """Possible ways to select basic block throughput from multiple values.

  Values:
    RANDOM: Picks a random value from the list of throughputs. All entries in
      the list have the same probability of being picked.
    FIRST: Picks the first value from the list of throughputs.
    MEAN: Computes the mean of the values in the list and uses it as the
      throughput of the block.
    MIN: Takes the minimal value in the list and uses it as the throughput of
      the block.
  """

  RANDOM = 0
  FIRST = 1
  MEAN = 2
  MIN = 3
