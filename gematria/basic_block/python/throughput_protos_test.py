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

"""Tests for throughput_protos."""

from absl.testing import absltest
from gematria.basic_block.python import basic_block
from gematria.basic_block.python import throughput_protos
from gematria.proto import basic_block_pb2
from gematria.proto import canonicalized_instruction_pb2
from gematria.proto import throughput_pb2


class ConvertFromProtoTest(absltest.TestCase):

  def test_empty_proto(self):
    proto = throughput_pb2.BasicBlockWithThroughputProto()
    block = throughput_protos.block_with_throughput_from_proto(proto)

    self.assertIsNotNone(block.block)
    self.assertEmpty(block.block.instructions)

  def test_proto_without_prefixes(self):
    proto = throughput_pb2.BasicBlockWithThroughputProto(
        basic_block=basic_block_pb2.BasicBlockProto(
            canonicalized_instructions=(
                canonicalized_instruction_pb2.CanonicalizedInstructionProto(
                    mnemonic='MOV',
                    llvm_mnemonic='MOV32rr',
                    input_operands=(
                        canonicalized_instruction_pb2.CanonicalizedOperandProto(
                            register_name='RAX'
                        ),
                    ),
                    output_operands=(
                        canonicalized_instruction_pb2.CanonicalizedOperandProto(
                            register_name='R15'
                        ),
                    ),
                ),
            ),
        ),
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                inverse_throughput_cycles=(1, 2, 3)
            ),
        ),
    )
    block = throughput_protos.block_with_throughput_from_proto(proto)

    self.assertEqual(
        block.block,
        basic_block.BasicBlock(
            basic_block.InstructionList((
                basic_block.Instruction(
                    mnemonic='MOV',
                    llvm_mnemonic='MOV32rr',
                    input_operands=basic_block.InstructionOperandList((
                        basic_block.InstructionOperand.from_register('RAX'),
                    )),
                    output_operands=basic_block.InstructionOperandList((
                        basic_block.InstructionOperand.from_register('R15'),
                    )),
                ),
            ))
        ),
    )

    self.assertLen(block.throughputs, 1)
    throughputs = block.throughputs[0]
    self.assertSequenceEqual(throughputs.inverse_throughput_cycles, (1, 2, 3))
    self.assertEmpty(throughputs.prefix_inverse_throughput_cycles)

  def test_proto_with_prefixes(self):
    proto = throughput_pb2.BasicBlockWithThroughputProto(
        basic_block=basic_block_pb2.BasicBlockProto(
            canonicalized_instructions=(
                canonicalized_instruction_pb2.CanonicalizedInstructionProto(
                    mnemonic='MOV',
                    llvm_mnemonic='MOV32rr',
                    input_operands=(
                        canonicalized_instruction_pb2.CanonicalizedOperandProto(
                            register_name='RAX'
                        ),
                    ),
                    output_operands=(
                        canonicalized_instruction_pb2.CanonicalizedOperandProto(
                            register_name='R15'
                        ),
                    ),
                ),
                canonicalized_instruction_pb2.CanonicalizedInstructionProto(
                    mnemonic='NOP', llvm_mnemonic='NOP'
                ),
            )
        ),
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                inverse_throughput_cycles=(1, 2, 3),
                prefix_inverse_throughputs=(
                    throughput_pb2.ThroughputWithSourceProto.PrefixThroughputProto(
                        inverse_throughput_cycles=(1, 1, 1)
                    ),
                    throughput_pb2.ThroughputWithSourceProto.PrefixThroughputProto(
                        inverse_throughput_cycles=(2, 2, 2)
                    ),
                ),
            ),
        ),
    )
    block = throughput_protos.block_with_throughput_from_proto(proto)

    self.assertEqual(
        block.block,
        basic_block.BasicBlock(
            basic_block.InstructionList((
                basic_block.Instruction(
                    mnemonic='MOV',
                    llvm_mnemonic='MOV32rr',
                    input_operands=basic_block.InstructionOperandList((
                        basic_block.InstructionOperand.from_register('RAX'),
                    )),
                    output_operands=basic_block.InstructionOperandList((
                        basic_block.InstructionOperand.from_register('R15'),
                    )),
                ),
                basic_block.Instruction(mnemonic='NOP', llvm_mnemonic='NOP'),
            ))
        ),
    )

    self.assertLen(block.throughputs, 1)
    throughputs = block.throughputs[0]
    self.assertSequenceEqual(throughputs.inverse_throughput_cycles, (1, 2, 3))
    self.assertSequenceEqual(
        throughputs.prefix_inverse_throughput_cycles, ((1, 1, 1), (2, 2, 2))
    )

  def test_proto_with_empty_sources(self):
    proto = throughput_pb2.BasicBlockWithThroughputProto(
        basic_block=basic_block_pb2.BasicBlockProto(
            canonicalized_instructions=(
                canonicalized_instruction_pb2.CanonicalizedInstructionProto(
                    mnemonic='MOV',
                    llvm_mnemonic='MOV32rr',
                    input_operands=(
                        canonicalized_instruction_pb2.CanonicalizedOperandProto(
                            register_name='RAX'
                        ),
                    ),
                    output_operands=(
                        canonicalized_instruction_pb2.CanonicalizedOperandProto(
                            register_name='R15'
                        ),
                    ),
                ),
                canonicalized_instruction_pb2.CanonicalizedInstructionProto(
                    mnemonic='NOP', llvm_mnemonic='NOP'
                ),
            )
        ),
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='source_1', inverse_throughput_cycles=(1, 2, 3)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='source_2', inverse_throughput_cycles=()
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='source_3', inverse_throughput_cycles=(4, 5, 6)
            ),
        ),
    )
    block = throughput_protos.block_with_throughput_from_proto(proto)

    self.assertEqual(
        block.block,
        basic_block.BasicBlock(
            basic_block.InstructionList((
                basic_block.Instruction(
                    mnemonic='MOV',
                    llvm_mnemonic='MOV32rr',
                    input_operands=basic_block.InstructionOperandList((
                        basic_block.InstructionOperand.from_register('RAX'),
                    )),
                    output_operands=basic_block.InstructionOperandList((
                        basic_block.InstructionOperand.from_register('R15'),
                    )),
                ),
                basic_block.Instruction(mnemonic='NOP', llvm_mnemonic='NOP'),
            ))
        ),
    )
    self.assertLen(block.throughputs, 3)
    self.assertSequenceEqual(
        block.throughputs[0].inverse_throughput_cycles, (1, 2, 3)
    )
    self.assertIsNone(block.throughputs[1])
    self.assertSequenceEqual(
        block.throughputs[2].inverse_throughput_cycles, (4, 5, 6)
    )


if __name__ == '__main__':
  absltest.main()
