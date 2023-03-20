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
"""Contains proto conversion functions for basic blocks with throughput."""

from gematria.basic_block.python import basic_block_protos
from gematria.basic_block.python import throughput
from gematria.proto import throughput_pb2


def block_with_throughput_from_proto(
    proto: throughput_pb2.BasicBlockWithThroughputProto,
) -> throughput.BasicBlockWithThroughput:
  """Converts a BasicBlockWithThroughputProto to BasicBlockWithThroughput."""
  throughputs = []
  for throughput_proto in proto.inverse_throughputs:
    inverse_throughput_cycles = throughput_proto.inverse_throughput_cycles
    if inverse_throughput_cycles:
      throughputs.append(
          throughput.BasicBlockThroughput(
              inverse_throughput_cycles=inverse_throughput_cycles
          )
      )
      if throughput_proto.prefix_inverse_throughputs:
        throughputs[-1].prefix_inverse_throughput_cycles = tuple(
            tuple(prefix.inverse_throughput_cycles)
            for prefix in throughput_proto.prefix_inverse_throughputs
        )
    else:
      throughputs.append(None)

  return throughput.BasicBlockWithThroughput(
      block=basic_block_protos.basic_block_from_proto(proto.basic_block),
      throughputs=throughputs,
  )
