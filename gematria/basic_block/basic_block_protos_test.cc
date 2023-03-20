// Copyright 2022 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gematria/basic_block/basic_block_protos.h"

#include "gematria/basic_block/basic_block.h"
#include "gematria/proto/basic_block.pb.h"
#include "gematria/proto/canonicalized_instruction.pb.h"
#include "gematria/testing/parse_proto.h"
#include "gtest/gtest.h"

namespace gematria {
namespace {

using ::gematria::AddressTuple;
using ::gematria::BasicBlock;
using ::gematria::CanonicalizedInstructionProto;
using ::gematria::CanonicalizedOperandProto;
using ::gematria::Instruction;
using ::gematria::InstructionOperand;
using ::gematria::OperandType;

TEST(AddressTupleFromProtoTest, AllFields) {
  const CanonicalizedOperandProto::AddressTuple proto = ParseTextProto(R"pb(
    base_register: 'RAX'
    index_register: 'RSI'
    displacement: -16
    scaling: 2
    segment: 'FS'
  )pb");
  AddressTuple address_tuple = AddressTupleFromProto(proto);
  EXPECT_EQ(address_tuple.base_register, "RAX");
  EXPECT_EQ(address_tuple.index_register, "RSI");
  EXPECT_EQ(address_tuple.displacement, -16);
  EXPECT_EQ(address_tuple.scaling, 2);
  EXPECT_EQ(address_tuple.segment_register, "FS");
}

TEST(InstructionOperandFromProtoTest, Register) {
  const CanonicalizedOperandProto proto =
      ParseTextProto("register_name: 'XMM0'");
  InstructionOperand operand = InstructionOperandFromProto(proto);
  EXPECT_EQ(operand, InstructionOperand::Register("XMM0"));

  EXPECT_EQ(operand.type(), OperandType::kRegister);
  EXPECT_EQ(operand.register_name(), "XMM0");
}

TEST(InstructionOperandFromProtoTest, ImmediateValue) {
  const CanonicalizedOperandProto proto = ParseTextProto("immediate_value: 12");
  InstructionOperand operand = InstructionOperandFromProto(proto);
  EXPECT_EQ(operand, InstructionOperand::ImmediateValue(12));
}

TEST(InstructionOperandFromProtoTest, FpImmediateValue) {
  const CanonicalizedOperandProto proto =
      ParseTextProto("fp_immediate_value: 1.234");
  InstructionOperand operand = InstructionOperandFromProto(proto);
  EXPECT_EQ(operand, InstructionOperand::FpImmediateValue(1.234));
}

TEST(InstructionOperandFromProtoTest, Address) {
  const CanonicalizedOperandProto proto = ParseTextProto(R"pb(
    address {
      base_register: 'RSI'
      index_register: 'RCX'
      displacement: 32
      scaling: 1
      segment: 'ES'
    })pb");
  InstructionOperand operand = InstructionOperandFromProto(proto);
  EXPECT_EQ(operand,
            InstructionOperand::Address(/* base_register = */ "RSI",
                                        /* displacement = */ 32,
                                        /* index_register = */ "RCX",
                                        /* scaling = */ 1,
                                        /* segment_register = */ "ES"));
}

TEST(InstructionOperandFromProtoTest, Memory) {
  const CanonicalizedOperandProto proto =
      ParseTextProto(R"pb(memory { alias_group_id: 123 })pb");
  InstructionOperand operand = InstructionOperandFromProto(proto);
  EXPECT_EQ(operand, InstructionOperand::MemoryLocation(123));
}

TEST(InstructionFromProtoTest, AllFields) {
  const CanonicalizedInstructionProto proto = ParseTextProto(R"pb(
    mnemonic: "ADC"
    prefixes: "LOCK"
    prefixes: "REP"
    llvm_mnemonic: "ADC32rr"
    output_operands { register_name: "RAX" }
    input_operands { register_name: "RAX" }
    input_operands { register_name: "RDI" }
    implicit_output_operands { register_name: "EFLAGS" }
    implicit_input_operands { register_name: "EFLAGS" }
    implicit_input_operands { immediate_value: 1 }
  )pb");
  Instruction instruction = InstructionFromProto(proto);
  EXPECT_EQ(
      instruction,
      Instruction(/* mnemonic = */ "ADC", /* llvm_mnemonic = */ "ADC32rr",
                  /* prefixes = */ {"LOCK", "REP"}, /* input_operands = */
                  {InstructionOperand::Register("RAX"),
                   InstructionOperand::Register("RDI")},
                  /* implicit_input_operands = */
                  {InstructionOperand::Register("EFLAGS"),
                   InstructionOperand::ImmediateValue(1)},
                  /* output_operands = */ {InstructionOperand::Register("RAX")},
                  /* implicit_output_operands = */
                  {InstructionOperand::Register("EFLAGS")}));
}

TEST(BasicBlockFromProtoTest, SomeInstructions) {
  const BasicBlockProto proto = ParseTextProto(R"pb(
    canonicalized_instructions: {
      mnemonic: "MOV"
      llvm_mnemonic: "MOV64rr"
      output_operands: { register_name: "RCX" }
      input_operands: { register_name: "RAX" }
    }
    canonicalized_instructions: {
      mnemonic: "NOT"
      llvm_mnemonic: "NOT64r"
      output_operands: { register_name: "RCX" }
      input_operands: { register_name: "RCX" }
    }
  )pb");
  const BasicBlock block = BasicBlockFromProto(proto);
  EXPECT_EQ(
      block,
      BasicBlock(
          {Instruction(
               /* mnemonic = */ "MOV", /* llvm_mnemonic = */ "MOV64rr",
               /* prefixes = */ {},
               /* input_operands = */ {InstructionOperand::Register("RAX")},
               /* implicit_input_operands = */ {},
               /* output_operands = */ {InstructionOperand::Register("RCX")},
               /* implicit_output_operands = */ {}),
           Instruction(
               /* mnemonic = */ "NOT",
               /* llvm_mnemonic = */ "NOT64r",
               /* prefixes = */ {},
               /* input_operands = */ {InstructionOperand::Register("RCX")},
               /* implicit_input_operands = */ {},
               /* output_operands = */ {InstructionOperand::Register("RCX")},
               /* implicit_output_operands = */ {})}));
}

}  // namespace
}  // namespace gematria
