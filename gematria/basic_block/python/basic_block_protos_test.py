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

"""Tests for basic basic block proto import functions."""

from absl.testing import absltest
from gematria.basic_block.python import basic_block
from gematria.basic_block.python import basic_block_protos
from gematria.proto import annotation_pb2
from gematria.proto import basic_block_pb2
from gematria.proto import canonicalized_instruction_pb2

_CanonicalizedOperandProto = (
    canonicalized_instruction_pb2.CanonicalizedOperandProto
)
_CanonicalizedInstructionProto = (
    canonicalized_instruction_pb2.CanonicalizedInstructionProto
)
_Annotation = (
    annotation_pb2.Annotation
)

class AddressTupleTest(absltest.TestCase):

  def test_address_tuple_from_proto(self):
    proto = _CanonicalizedOperandProto.AddressTuple(
        base_register='RCX',
        displacement=-1,
        index_register='RSI',
        scaling=2,
        segment='FS',
    )
    address = basic_block_protos.address_tuple_from_proto(proto)
    self.assertEqual(address.base_register, 'RCX')
    self.assertEqual(address.displacement, -1)
    self.assertEqual(address.index_register, 'RSI')
    self.assertEqual(address.scaling, 2)
    self.assertEqual(address.segment_register, 'FS')


class InstructionOperandFromProtoTest(absltest.TestCase):

  def test_register(self):
    proto = _CanonicalizedOperandProto(register_name='RBX')
    operand = basic_block_protos.instruction_operand_from_proto(proto)

    self.assertEqual(operand.type, basic_block.OperandType.REGISTER)
    self.assertEqual(operand.register_name, 'RBX')

    self.assertIsNone(operand.immediate_value)
    self.assertIsNone(operand.fp_immediate_value)
    self.assertIsNone(operand.address)
    self.assertIsNone(operand.alias_group_id)

  def test_immediate_value(self):
    proto = _CanonicalizedOperandProto(immediate_value=1234)
    operand = basic_block_protos.instruction_operand_from_proto(proto)

    self.assertEqual(operand.type, basic_block.OperandType.IMMEDIATE_VALUE)
    self.assertEqual(operand.immediate_value, 1234)

    self.assertIsNone(operand.register_name)
    self.assertIsNone(operand.fp_immediate_value)
    self.assertIsNone(operand.address)
    self.assertIsNone(operand.alias_group_id)

  def test_fp_imemdiate_value(self):
    proto = _CanonicalizedOperandProto(fp_immediate_value=9.999)
    operand = basic_block_protos.instruction_operand_from_proto(proto)

    self.assertEqual(operand.type, basic_block.OperandType.FP_IMMEDIATE_VALUE)
    self.assertEqual(operand.fp_immediate_value, 9.999)

    self.assertIsNone(operand.register_name)
    self.assertIsNone(operand.immediate_value)
    self.assertIsNone(operand.address)
    self.assertIsNone(operand.alias_group_id)

  def test_address(self):
    proto = _CanonicalizedOperandProto(
        address=_CanonicalizedOperandProto.AddressTuple(
            base_register='RBX', displacement=-8
        )
    )
    operand = basic_block_protos.instruction_operand_from_proto(proto)

    self.assertEqual(operand.type, basic_block.OperandType.ADDRESS)
    self.assertEqual(operand.address.base_register, 'RBX')
    self.assertEqual(operand.address.displacement, -8)

    self.assertIsNone(operand.register_name)
    self.assertIsNone(operand.immediate_value)
    self.assertIsNone(operand.fp_immediate_value)
    self.assertIsNone(operand.alias_group_id)

  def test_memory(self):
    proto = _CanonicalizedOperandProto(
        memory=_CanonicalizedOperandProto.MemoryLocation(alias_group_id=123)
    )
    operand = basic_block_protos.instruction_operand_from_proto(proto)

    self.assertEqual(operand.type, basic_block.OperandType.MEMORY)
    self.assertEqual(operand.alias_group_id, 123)

    self.assertIsNone(operand.register_name)
    self.assertIsNone(operand.immediate_value)
    self.assertIsNone(operand.fp_immediate_value)
    self.assertIsNone(operand.address)


class InstructionFromProtoTest(absltest.TestCase):

  def test_instruction_from_proto(self):
    proto = _CanonicalizedInstructionProto(
        mnemonic='ADC',
        llvm_mnemonic='ADC32rr',
        prefixes=['LOCK'],
        input_operands=(
            _CanonicalizedOperandProto(register_name='RAX'),
            _CanonicalizedOperandProto(register_name='RBX'),
        ),
        implicit_output_operands=(
            _CanonicalizedOperandProto(register_name='EFLAGS'),
        ),
        output_operands=(_CanonicalizedOperandProto(register_name='RBX'),),
        implicit_input_operands=(
            _CanonicalizedOperandProto(register_name='EFLAGS'),
        ),
    )
    instruction = basic_block_protos.instruction_from_proto(proto)
    self.assertEqual(instruction.mnemonic, 'ADC')
    self.assertEqual(instruction.llvm_mnemonic, 'ADC32rr')
    self.assertSequenceEqual(instruction.prefixes, ('LOCK',))
    self.assertSequenceEqual(
        instruction.input_operands,
        (
            basic_block.InstructionOperand.from_register('RAX'),
            basic_block.InstructionOperand.from_register('RBX'),
        ),
    )
    self.assertSequenceEqual(
        instruction.implicit_input_operands,
        (basic_block.InstructionOperand.from_register('EFLAGS'),),
    )
    self.assertSequenceEqual(
        instruction.output_operands,
        (basic_block.InstructionOperand.from_register('RBX'),),
    )
    self.assertSequenceEqual(
        instruction.implicit_output_operands,
        (basic_block.InstructionOperand.from_register('EFLAGS'),),
    )


class BasicBlockFromProtoTest(absltest.TestCase):

  def test_initialize_from_proto(self):
    proto = basic_block_pb2.BasicBlockProto(
        canonicalized_instructions=(
            _CanonicalizedInstructionProto(
                mnemonic='MOV',
                llvm_mnemonic='MOV32rr',
                input_operands=(
                    _CanonicalizedOperandProto(register_name='RSI'),
                ),
                output_operands=(
                    _CanonicalizedOperandProto(register_name='RCX'),
                ),
                instruction_annotations=(
                    _Annotation(
                        name='MEM_LOAD_RETIRED:L3_MISS',
                        value=0.875,
                    ),
                ),
            ),
            _CanonicalizedInstructionProto(
                mnemonic='MOVSB',
                llvm_mnemonic='MOVSB',
                implicit_input_operands=(
                    _CanonicalizedOperandProto(register_name='RCX'),
                    _CanonicalizedOperandProto(register_name='RSI'),
                    _CanonicalizedOperandProto(register_name='RDI'),
                    _CanonicalizedOperandProto(
                        memory=_CanonicalizedOperandProto.MemoryLocation(
                            alias_group_id=1
                        )
                    ),
                ),
                implicit_output_operands=(
                    _CanonicalizedOperandProto(register_name='RSI'),
                    _CanonicalizedOperandProto(register_name='RDI'),
                    _CanonicalizedOperandProto(
                        memory=_CanonicalizedOperandProto.MemoryLocation(
                            alias_group_id=2
                        )
                    ),
                ),
                instruction_annotations=(
                    _Annotation(
                        name='MEM_LOAD_RETIRED:L3_MISS',
                        value=0.95,
                    ),
                ),
            ),
        )
    )
    block = basic_block_protos.basic_block_from_proto(proto)

    expected = (
        basic_block.Instruction(
            mnemonic='MOV',
            llvm_mnemonic='MOV32rr',
            input_operands=basic_block.InstructionOperandList((
                basic_block.InstructionOperand.from_register('RSI'),
            )),
            output_operands=basic_block.InstructionOperandList((
                basic_block.InstructionOperand.from_register('RCX'),
            )),
            instruction_annotations=basic_block.AnnotationList((
                basic_block.Annotation(
                    name='MEM_LOAD_RETIRED:L3_MISS',
                    value=0.875,
                ),
            )),
        ),
        basic_block.Instruction(
            mnemonic='MOVSB',
            llvm_mnemonic='MOVSB',
            implicit_input_operands=basic_block.InstructionOperandList((
                basic_block.InstructionOperand.from_register('RCX'),
                basic_block.InstructionOperand.from_register('RSI'),
                basic_block.InstructionOperand.from_register('RDI'),
                basic_block.InstructionOperand.from_memory(1),
            )),
            implicit_output_operands=basic_block.InstructionOperandList((
                basic_block.InstructionOperand.from_register('RSI'),
                basic_block.InstructionOperand.from_register('RDI'),
                basic_block.InstructionOperand.from_memory(2),
            )),
            instruction_annotations=basic_block.AnnotationList((
                basic_block.Annotation(
                    name='MEM_LOAD_RETIRED:L3_MISS',
                    value=0.95,
                ),
            )),
        ),
    )
    self.assertSequenceEqual(block.instructions, expected)


if __name__ == '__main__':
  absltest.main()
