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
"""Tests for basic block data structure wrappers."""

from absl.testing import absltest

from gematria.basic_block.python import basic_block
from gematria.basic_block.python import tokens


class OperandTypeTest(absltest.TestCase):

  def test_values(self):
    self.assertGreaterEqual(len(basic_block.OperandType.__members__), 0)

  def test_docstring(self):
    docstring = basic_block.OperandType.__doc__
    for value in basic_block.OperandType.__members__:
      self.assertIn(value, docstring)


class AddressTupleTest(absltest.TestCase):

  def test_initialize(self):
    address = basic_block.AddressTuple()
    self.assertEqual(address.base_register, '')
    self.assertEqual(address.displacement, 0)
    self.assertEqual(address.index_register, '')
    self.assertEqual(address.scaling, 0)
    self.assertEqual(address.segment_register, '')

    # The attributes of an address tuple are immutable, so that we don't get any
    # unexpected behavior on the C++/Python frontier.
    with self.assertRaises(AttributeError):
      address.base_register = 'RSI'
    with self.assertRaises(AttributeError):
      address.displacement = 123
    with self.assertRaises(AttributeError):
      address.index_register = 'RSI'
    with self.assertRaises(AttributeError):
      address.scaling = 1
    with self.assertRaises(AttributeError):
      address.segment_register = 'FS'

  def test_initialize_with_keyword_args(self):
    address = basic_block.AddressTuple(base_register='RCX', displacement=12345)
    self.assertEqual(address.base_register, 'RCX')
    self.assertEqual(address.displacement, 12345)
    self.assertEqual(address.index_register, '')
    self.assertEqual(address.scaling, 0)
    self.assertEqual(address.segment_register, '')

    address = basic_block.AddressTuple(
        index_register='R12', scaling=2, segment_register='FS')
    self.assertEqual(address.base_register, '')
    self.assertEqual(address.displacement, 0)
    self.assertEqual(address.index_register, 'R12')
    self.assertEqual(address.scaling, 2)
    self.assertEqual(address.segment_register, 'FS')

  def test_equality(self):
    address_1 = basic_block.AddressTuple(base_register='RCX', displacement=123)
    self.assertEqual(address_1, address_1)

    address_1a = basic_block.AddressTuple(base_register='RCX', displacement=123)
    self.assertEqual(address_1, address_1a)

    address_2 = basic_block.AddressTuple(
        base_register='RCX', displacement=123, index_register='RDI')
    self.assertEqual(address_2, address_2)
    self.assertNotEqual(address_1, address_2)

  def test_repr(self):
    address = basic_block.AddressTuple()
    self.assertEqual(repr(address), 'AddressTuple()')
    self.assertEqual(str(address), 'AddressTuple()')

    address = basic_block.AddressTuple(base_register='RAX')
    self.assertEqual(repr(address), "AddressTuple(base_register='RAX')")
    self.assertEqual(str(address), "AddressTuple(base_register='RAX')")


class InstructionOperandTest(absltest.TestCase):

  def test_initialize_register(self):
    operand = basic_block.InstructionOperand.from_register(register_name='RCX')
    self.assertEqual(operand.type, basic_block.OperandType.REGISTER)
    self.assertEqual(operand.register_name, 'RCX')

    self.assertIsNone(operand.immediate_value)
    self.assertIsNone(operand.fp_immediate_value)
    self.assertIsNone(operand.address)
    self.assertIsNone(operand.alias_group_id)

  def test_initialize_immediate(self):
    operand = basic_block.InstructionOperand.from_immediate_value(
        immediate_value=321)
    self.assertEqual(operand.type, basic_block.OperandType.IMMEDIATE_VALUE)
    self.assertEqual(operand.immediate_value, 321)

    self.assertIsNone(operand.register_name)
    self.assertIsNone(operand.fp_immediate_value)
    self.assertIsNone(operand.address)
    self.assertIsNone(operand.alias_group_id)

  def test_initialize_fp_immediate(self):
    operand = basic_block.InstructionOperand.from_fp_immediate_value(
        fp_immediate_value=1.23)
    self.assertEqual(operand.type, basic_block.OperandType.FP_IMMEDIATE_VALUE)
    self.assertEqual(operand.fp_immediate_value, 1.23)

    self.assertIsNone(operand.register_name)
    self.assertIsNone(operand.immediate_value)
    self.assertIsNone(operand.address)
    self.assertIsNone(operand.alias_group_id)

  def test_initialize_address(self):
    operand = basic_block.InstructionOperand.from_address(
        address_tuple=basic_block.AddressTuple(
            base_register='RCX', displacement=-8))
    self.assertEqual(operand.type, basic_block.OperandType.ADDRESS)
    self.assertEqual(operand.address.base_register, 'RCX')
    self.assertEqual(operand.address.displacement, -8)

    self.assertIsNone(operand.register_name)
    self.assertIsNone(operand.immediate_value)
    self.assertIsNone(operand.fp_immediate_value)
    self.assertIsNone(operand.alias_group_id)

  def test_initialize_memory(self):
    operand = basic_block.InstructionOperand.from_memory(alias_group_id=123)
    self.assertEqual(operand.type, basic_block.OperandType.MEMORY)
    self.assertEqual(operand.alias_group_id, 123)

    self.assertIsNone(operand.register_name)
    self.assertIsNone(operand.immediate_value)
    self.assertIsNone(operand.fp_immediate_value)
    self.assertIsNone(operand.address)

  def test_as_token_list(self):
    test_cases = (
        (basic_block.InstructionOperand.from_register('RBX'), ('RBX',)),
        (
            basic_block.InstructionOperand.from_immediate_value(123),
            (tokens.IMMEDIATE,),
        ),
        (
            basic_block.InstructionOperand.from_fp_immediate_value(1.23),
            (tokens.IMMEDIATE,),
        ),
        (
            basic_block.InstructionOperand.from_address(
                base_register='RCX', index_register='RDX', displacement=-8),
            (tokens.ADDRESS, 'RCX', 'RDX', tokens.DISPLACEMENT),
        ),
        (
            basic_block.InstructionOperand.from_memory(alias_group_id=2),
            (tokens.MEMORY,),
        ),
    )
    for operand, expected_token_list in test_cases:
      self.assertSequenceEqual(operand.as_token_list(), expected_token_list,
                               f'operand = {operand!r}')


class InstructionTest(absltest.TestCase):

  def test_initialize_with_keyword_args(self):
    instruction = basic_block.Instruction(
        mnemonic='ADC',
        llvm_mnemonic='ADC32rr',
        input_operands=basic_block.InstructionOperandList((
            basic_block.InstructionOperand.from_register('RCX'),
            basic_block.InstructionOperand.from_register('RDX'),
        )),
        implicit_input_operands=basic_block.InstructionOperandList(
            (basic_block.InstructionOperand.from_register('EFLAGS'),)),
        output_operands=basic_block.InstructionOperandList(
            (basic_block.InstructionOperand.from_register('RDX'),)),
        implicit_output_operands=basic_block.InstructionOperandList(
            (basic_block.InstructionOperand.from_register('EFLAGS'),)),
    )
    self.assertEqual(instruction.mnemonic, 'ADC')
    self.assertEqual(instruction.llvm_mnemonic, 'ADC32rr')
    self.assertEmpty(instruction.prefixes)
    self.assertSequenceEqual(
        instruction.input_operands,
        (
            basic_block.InstructionOperand.from_register('RCX'),
            basic_block.InstructionOperand.from_register('RDX'),
        ),
    )
    self.assertSequenceEqual(
        instruction.implicit_input_operands,
        (basic_block.InstructionOperand.from_register('EFLAGS'),),
    )
    self.assertSequenceEqual(
        instruction.output_operands,
        (basic_block.InstructionOperand.from_register('RDX'),),
    )
    self.assertSequenceEqual(
        instruction.implicit_output_operands,
        (basic_block.InstructionOperand.from_register('EFLAGS'),),
    )

    instruction = basic_block.Instruction(
        mnemonic='NOP', prefixes=basic_block.StringList(('LOCK',)))
    self.assertEqual(instruction.mnemonic, 'NOP')
    self.assertEqual(instruction.llvm_mnemonic, '')
    self.assertSequenceEqual(instruction.prefixes, ('LOCK',))
    self.assertEmpty(instruction.input_operands)
    self.assertEmpty(instruction.implicit_input_operands)
    self.assertEmpty(instruction.output_operands)
    self.assertEmpty(instruction.implicit_output_operands)

  def test_equality(self):
    instruction_a1 = basic_block.Instruction(mnemonic='NOP')
    instruction_a2 = basic_block.Instruction(mnemonic='NOP')
    instruction_a3 = basic_block.Instruction(
        mnemonic='NOP', llvm_mnemonic='NOP')
    self.assertEqual(instruction_a1, instruction_a1)
    self.assertEqual(instruction_a1, instruction_a2)
    # The two instructions differ by their llvm_mnemonic.
    self.assertNotEqual(instruction_a1, instruction_a3)

    instruction_b1 = basic_block.Instruction(
        mnemonic='MOV',
        llvm_mnemonic='MOV32rrr',
        input_operands=basic_block.InstructionOperandList(
            (basic_block.InstructionOperand.from_register('RCX'),)),
        output_operands=basic_block.InstructionOperandList(
            (basic_block.InstructionOperand.from_register('RDX'),)),
    )
    instruction_b2 = basic_block.Instruction(
        mnemonic='MOV',
        llvm_mnemonic='MOV32rrr',
        input_operands=basic_block.InstructionOperandList(
            (basic_block.InstructionOperand.from_register('RCX'),)),
        output_operands=basic_block.InstructionOperandList(
            (basic_block.InstructionOperand.from_register('RDX'),)),
    )
    instruction_b3 = basic_block.Instruction(
        mnemonic='MOV',
        llvm_mnemonic='MOV32rrr',
        input_operands=basic_block.InstructionOperandList(
            (basic_block.InstructionOperand.from_register('RSI'),)),
        output_operands=basic_block.InstructionOperandList(
            (basic_block.InstructionOperand.from_register('RDI'),)),
    )
    self.assertEqual(instruction_b1, instruction_b2)
    self.assertNotEqual(instruction_a1, instruction_b2)
    self.assertNotEqual(instruction_b1, instruction_b3)

  def test_modify_prefixes(self):
    instruction = basic_block.Instruction()
    instruction.prefixes.append('REP')
    self.assertSequenceEqual(instruction.prefixes, ('REP',))

    instruction.prefixes = basic_block.StringList(('REP', 'LOCK'))
    self.assertSequenceEqual(instruction.prefixes, ['REP', 'LOCK'])

    instruction.prefixes[0] = 'REPNE'
    self.assertSequenceEqual(instruction.prefixes, ['REPNE', 'LOCK'])

  def test_modify_input_operands(self):
    instruction = basic_block.Instruction()
    instruction.input_operands.append(
        basic_block.InstructionOperand.from_register('RBX'))
    self.assertSequenceEqual(
        instruction.input_operands,
        (basic_block.InstructionOperand.from_register('RBX'),),
    )

    instruction.input_operands = basic_block.InstructionOperandList()
    self.assertEmpty(instruction.input_operands)

  def test_modify_output_operands(self):
    instruction = basic_block.Instruction()
    instruction.output_operands.append(
        basic_block.InstructionOperand.from_register('RCX'))
    self.assertSequenceEqual(
        instruction.output_operands,
        (basic_block.InstructionOperand.from_register('RCX'),),
    )

    instruction.output_operands = basic_block.InstructionOperandList()
    self.assertEmpty(instruction.output_operands)

  def test_as_token_list(self):
    test_cases = ((
        basic_block.Instruction(
            mnemonic='ADD',
            llvm_mnemonic='ADD32rr',
            prefixes=basic_block.StringList(('LOCK',)),
            input_operands=basic_block.InstructionOperandList((
                basic_block.InstructionOperand.from_register('RAX'),
                basic_block.InstructionOperand.from_register('RBX'),
            )),
            implicit_input_operands=basic_block.InstructionOperandList(
                (basic_block.InstructionOperand.from_register('EFLAGS'),)),
            output_operands=basic_block.InstructionOperandList(
                (basic_block.InstructionOperand.from_register('RBX'),)),
            implicit_output_operands=basic_block.InstructionOperandList(
                (basic_block.InstructionOperand.from_register('EFLAGS'),)),
        ),
        (
            'LOCK',
            'ADD',
            tokens.DELIMITER,
            'RBX',
            'EFLAGS',
            tokens.DELIMITER,
            'RAX',
            'RBX',
            'EFLAGS',
            tokens.DELIMITER,
        ),
    ),)
    for instruction, expected_token_list in test_cases:
      token_list = instruction.as_token_list()
      self.assertSequenceEqual(token_list, expected_token_list)


class BasicBlockTest(absltest.TestCase):

  def test_equality(self):
    block_1a = basic_block.BasicBlock(
        instructions=basic_block.InstructionList((
            basic_block.Instruction(
                mnemonic='MOV',
                llvm_mnemonic='MOV32rr',
                input_operands=basic_block.InstructionOperandList((
                    basic_block.InstructionOperand.from_register('RSI'),)),
                output_operands=basic_block.InstructionOperandList((
                    basic_block.InstructionOperand.from_register('RCX'),)),
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
            ),
        )))
    block_1b = basic_block.BasicBlock(
        instructions=basic_block.InstructionList((
            basic_block.Instruction(
                mnemonic='MOV',
                llvm_mnemonic='MOV32rr',
                input_operands=basic_block.InstructionOperandList((
                    basic_block.InstructionOperand.from_register('RSI'),)),
                output_operands=basic_block.InstructionOperandList((
                    basic_block.InstructionOperand.from_register('RCX'),)),
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
            ),
        )))
    self.assertEqual(block_1a, block_1a)
    self.assertEqual(block_1a, block_1b)

    block_2 = basic_block.BasicBlock(
        instructions=basic_block.InstructionList((basic_block.Instruction(
            mnemonic='MOV',
            llvm_mnemonic='MOV32rr',
            input_operands=basic_block.InstructionOperandList((
                basic_block.InstructionOperand.from_register('RSI'),)),
            output_operands=basic_block.InstructionOperandList((
                basic_block.InstructionOperand.from_register('RCX'),)),
        ),)))
    self.assertNotEqual(block_1a, block_2)

  def test_modify_instruction_list(self):
    instructions = (
        basic_block.Instruction(
            mnemonic='MOV',
            llvm_mnemonic='MOV32rr',
            input_operands=basic_block.InstructionOperandList(
                (basic_block.InstructionOperand.from_register('RSI'),)),
            output_operands=basic_block.InstructionOperandList(
                (basic_block.InstructionOperand.from_register('RCX'),)),
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
        ),
    )

    block = basic_block.BasicBlock(
        instructions=basic_block.InstructionList(instructions))
    self.assertSequenceEqual(block.instructions, instructions)

    del block.instructions[1]
    self.assertSequenceEqual(block.instructions, (instructions[0],))

    block.instructions.extend(instructions)
    self.assertSequenceEqual(
        block.instructions, (instructions[0], instructions[0], instructions[1]))

    # Make sure that the property does not return a copy of the list.
    block.instructions[0].mnemonic = 'ADD'
    self.assertEqual(block.instructions[0].mnemonic, 'ADD')


if __name__ == '__main__':
  absltest.main()
