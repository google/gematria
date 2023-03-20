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

"""Contains classes that hold basic blocks along with throughput information."""

from collections.abc import Sequence
import dataclasses
from typing import Optional

from gematria.basic_block.python import basic_block


@dataclasses.dataclass
class BasicBlockThroughput:
  """Contains throughput information for a single basic block and task.

  Attributes:
    inverse_throughput_cycles: The number of cycles of inverse throughput of the
      basic block. Each element of the sequence corresponds to a single
      measurement of the inverse throughput.
    prefix_inverse_throughput_cycles: The number of cycles of inverse throughput
      of the prefixes of the basic block. The outer sequence corresponds to the
      prefixes of the basic block; when non-empty, it must have the same number
      of elements as there are instructions in the basic block. The inner
      sequence corresponds to different measurements on the prefix of the basic
      block.
  """

  inverse_throughput_cycles: Sequence[float]
  prefix_inverse_throughput_cycles: Sequence[Sequence[float]] = ()


@dataclasses.dataclass
class BasicBlockWithThroughput:
  """Contains basic block definition along with throughput information.

  Attributes:
    block: The basic block definition.
    throughputs: The inverse throughputs for the basic block. Each entry of the
      sequence corresponds to one task in the model (one microarchitecture to
      predict).
  """

  block: basic_block.BasicBlock
  throughputs: Sequence[Optional[BasicBlockThroughput]] = ()
