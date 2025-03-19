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

#include "gematria/basic_block/basic_block.h"

#include <cstdint>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace gematria {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

TEST(AddressTupleTest, Constructor) {
  const std::string kBaseRegister = "RAX";
  const int64_t kDisplacement = 1234;
  const std::string kIndexRegister = "RBX";
  const int kScaling = 2;
  const std::string kSegmentRegister = "FS";

  const AddressTuple address(/* base_register = */ kBaseRegister,
                             /* displacement = */ kDisplacement,
                             /* index_register = */ kIndexRegister,
                             /* scaling = */ kScaling,
                             /* segment_register = */ kSegmentRegister);
  EXPECT_EQ(address.base_register, kBaseRegister);
  EXPECT_EQ(address.displacement, kDisplacement);
  EXPECT_EQ(address.index_register, kIndexRegister);
  EXPECT_EQ(address.scaling, kScaling);
  EXPECT_EQ(address.segment_register, kSegmentRegister);
}

TEST(AddressTupleTest, ToString) {
  const struct {
    AddressTuple address;
    const char* expected_string;
  } kTestCases[] = {
      {AddressTuple(/* base_register = */ "RAX",
                    /* displacement = */ 0,
                    /* index_register = */ "",
                    /* scaling = */ 0,
                    /* segment_register = */ ""),
       "AddressTuple(base_register='RAX')"},
      {AddressTuple(/* base_register = */ "RAX",
                    /* displacement = */ 16,
                    /* index_register = */ "",
                    /* scaling = */ 0,
                    /* segment_register = */ ""),
       "AddressTuple(base_register='RAX', displacement=16)"},
      {AddressTuple(/* base_register = */ "RAX",
                    /* displacement = */ 0,
                    /* index_register = */ "RSI",
                    /* scaling = */ 0,
                    /* segment_register = */ ""),
       "AddressTuple(base_register='RAX', index_Register='RSI', scaling=0)"},
      {AddressTuple(/* base_register = */ "RAX",
                    /* displacement = */ -8,
                    /* index_register = */ "RSI",
                    /* scaling = */ 1,
                    /* segment_register = */ ""),
       "AddressTuple(base_register='RAX', displacement=-8, "
       "index_Register='RSI', scaling=1)"},
      {AddressTuple(/* base_register = */ "RAX",
                    /* displacement = */ -123,
                    /* index_register = */ "RSI",
                    /* scaling = */ 1,
                    /* segment_register = */ "ES"),
       "AddressTuple(base_register='RAX', displacement=-123, "
       "index_Register='RSI', scaling=1, segment_register='ES')"}};

  for (const auto& test_case : kTestCases) {
    SCOPED_TRACE(test_case.expected_string);
    EXPECT_EQ(test_case.address.ToString(), test_case.expected_string);
  }
}

TEST(AddressTupleTest, Equality) {
  AddressTuple address_1;
  AddressTuple address_2;

  // Both are empty.
  EXPECT_EQ(address_1, address_2);

  // Set base register on one.
  address_1.base_register = "RAX";
  EXPECT_NE(address_1, address_2);
  EXPECT_FALSE(address_1 == address_2);

  // Set the same base on both.
  address_2.base_register = "RAX";
  EXPECT_EQ(address_1, address_2);
  EXPECT_FALSE(address_1 != address_2);

  // Set different base on each.
  address_1.base_register = "RSI";
  EXPECT_NE(address_1, address_2);
  EXPECT_FALSE(address_1 == address_2);

  // Set same base, different displacement on each.
  address_2.base_register = "RSI";
  address_2.displacement = -32;
  EXPECT_NE(address_1, address_2);
  EXPECT_FALSE(address_1 == address_2);

  // Set same displacement on both.
  address_1.displacement = -32;
  EXPECT_EQ(address_1, address_2);
  EXPECT_FALSE(address_1 != address_2);

  // Set different index register on each.
  address_1.index_register = "RDI";
  EXPECT_NE(address_1, address_2);
  EXPECT_FALSE(address_1 == address_2);

  // Set same index register on both.
  address_2.index_register = "RDI";
  EXPECT_EQ(address_1, address_2);
  EXPECT_FALSE(address_1 != address_2);

  // Set different scaling on each.
  address_1.scaling = 3;
  EXPECT_NE(address_1, address_2);
  EXPECT_FALSE(address_1 == address_2);

  // Set same scaling on both.
  address_2.scaling = 3;
  EXPECT_EQ(address_1, address_2);
  EXPECT_FALSE(address_1 != address_2);

  // Set different segment register on each.
  address_1.segment_register = "FS";
  EXPECT_NE(address_1, address_2);
  EXPECT_FALSE(address_1 == address_2);

  // Set same segment register on each.
  address_2.segment_register = "FS";
  EXPECT_EQ(address_1, address_2);
  EXPECT_FALSE(address_1 != address_2);
}

TEST(InstructionOperandTest, ConstructorEmpty) {
  const InstructionOperand opreand;
  EXPECT_EQ(opreand.type(), OperandType::kUnknown);
}

TEST(InstructionOperandTest, ConstructorRegister) {
  const auto operand = InstructionOperand::Register("R10");
  EXPECT_EQ(operand.type(), OperandType::kRegister);
  EXPECT_EQ(operand.register_name(), "R10");
}

TEST(InstructionOperandTest, ConstructorImmediateValue) {
  const auto operand = InstructionOperand::ImmediateValue(123);
  EXPECT_EQ(operand.type(), OperandType::kImmediateValue);
  EXPECT_EQ(operand.immediate_value(), 123);
}

TEST(InstructionOperandTest, ConstructorFpImmediateValue) {
  const double kValue = 3.14;
  const auto operand = InstructionOperand::FpImmediateValue(kValue);
  EXPECT_EQ(operand.type(), OperandType::kFpImmediateValue);
  EXPECT_EQ(operand.fp_immediate_value(), kValue);
}

TEST(InstructionOperandTest, ConstructorAddressFromTuple) {
  const AddressTuple address("RSI", -16, "RDI", 0, "");
  const auto operand = InstructionOperand::Address(address);
  EXPECT_EQ(operand.type(), OperandType::kAddress);
  EXPECT_EQ(operand.address(), address);
}

TEST(InstructionOperandTest, ConstructorAddressFromArguments) {
  const auto operand = InstructionOperand::Address("RSI", -16, "RDI", 0, "");
  EXPECT_EQ(operand.type(), OperandType::kAddress);
  EXPECT_EQ(operand.address(), AddressTuple("RSI", -16, "RDI", 0, ""));
}

TEST(InstructionOperandTest, ConstructorFromMemoryLocation) {
  const int kAliasGroupId = 42;
  const auto operand = InstructionOperand::MemoryLocation(42);
  EXPECT_EQ(operand.type(), OperandType::kMemory);
  EXPECT_EQ(operand.alias_group_id(), kAliasGroupId);
}

TEST(InstructionOperandTest, Equality) {
  // Registers.
  const auto operand_rax_a = InstructionOperand::Register("RAX");
  const auto operand_rax_b = InstructionOperand::Register("RAX");
  EXPECT_EQ(operand_rax_a, operand_rax_b);
  EXPECT_FALSE(operand_rax_a != operand_rax_b);

  const auto operand_rbx = InstructionOperand::Register("RBX");
  EXPECT_NE(operand_rax_a, operand_rbx);
  EXPECT_FALSE(operand_rax_a == operand_rbx);

  // Immediate values.
  const auto value_1_a = InstructionOperand::ImmediateValue(1);
  const auto value_1_b = InstructionOperand::ImmediateValue(1);
  EXPECT_EQ(value_1_a, value_1_b);
  EXPECT_FALSE(value_1_a != value_1_b);

  const auto value_2 = InstructionOperand::ImmediateValue(2);
  EXPECT_NE(value_1_a, value_2);
  EXPECT_FALSE(value_1_a == value_2);

  EXPECT_NE(operand_rax_a, value_1_a);
  EXPECT_FALSE(operand_rax_a == value_1_a);

  // FP immediate values.
  const auto fp_value_1_a = InstructionOperand::FpImmediateValue(1.0);
  const auto fp_value_1_b = InstructionOperand::FpImmediateValue(1.0);
  EXPECT_EQ(fp_value_1_a, fp_value_1_b);
  EXPECT_FALSE(fp_value_1_a != fp_value_1_b);

  const auto fp_value_2 = InstructionOperand::FpImmediateValue(2.0);
  EXPECT_NE(fp_value_1_a, fp_value_2);
  EXPECT_FALSE(fp_value_1_a == fp_value_2);

  EXPECT_NE(operand_rax_a, fp_value_1_a);
  EXPECT_FALSE(operand_rax_a == fp_value_1_a);
  EXPECT_NE(value_1_a, fp_value_1_a);
  EXPECT_FALSE(value_1_a == fp_value_1_a);

  // Address.
  const auto address_rax_a = InstructionOperand::Address("RAX", 0, "", 0, "");
  const auto address_rax_b = InstructionOperand::Address("RAX", 0, "", 0, "");
  EXPECT_EQ(address_rax_a, address_rax_b);
  EXPECT_FALSE(address_rax_a != address_rax_b);

  const auto address_rax_rbx =
      InstructionOperand::Address("RAX", 0, "RBX", 0, "");
  EXPECT_NE(address_rax_a, address_rax_rbx);
  EXPECT_FALSE(address_rax_a == address_rax_rbx);

  EXPECT_NE(operand_rax_a, address_rax_a);
  EXPECT_FALSE(operand_rax_a == address_rax_a);
  EXPECT_NE(value_1_a, address_rax_a);
  EXPECT_FALSE(value_1_a == address_rax_a);
  EXPECT_NE(fp_value_1_a, address_rax_a);
  EXPECT_FALSE(fp_value_1_a == address_rax_a);

  // Memory.
  const auto memory_1_a = InstructionOperand::MemoryLocation(1);
  const auto memory_1_b = InstructionOperand::MemoryLocation(1);
  EXPECT_EQ(memory_1_a, memory_1_b);
  EXPECT_FALSE(memory_1_a != memory_1_b);

  const auto memory_2 = InstructionOperand::MemoryLocation(2);
  EXPECT_NE(memory_1_a, memory_2);
  EXPECT_FALSE(memory_1_a == memory_2);

  EXPECT_NE(operand_rax_a, memory_1_a);
  EXPECT_FALSE(operand_rax_a == memory_1_a);
  EXPECT_NE(value_1_a, memory_1_a);
  EXPECT_FALSE(value_1_a == memory_1_a);
  EXPECT_NE(fp_value_1_a, memory_1_a);
  EXPECT_FALSE(fp_value_1_a == memory_1_a);
  EXPECT_NE(address_rax_a, memory_1_a);
  EXPECT_FALSE(address_rax_a == memory_1_a);
}

TEST(InstructionOperandTest, ToString) {
  const struct {
    InstructionOperand operand;
    const char* expected_string;
  } kTestCases[] = {
      {InstructionOperand::Register("RAX"),
       "InstructionOperand.from_register('RAX')"},
      {InstructionOperand::ImmediateValue(333),
       "InstructionOperand.from_immediate_value(333)"},
      {InstructionOperand::FpImmediateValue(3.14),
       "InstructionOperand.from_fp_immediate_value(3.14)"},
      {InstructionOperand::Address("RAX", 0, "", 0, ""),
       "InstructionOperand.from_address(AddressTuple(base_register='RAX'))"},
      {InstructionOperand::MemoryLocation(32),
       "InstructionOperand.from_memory(32)"}};

  for (const auto& test_case : kTestCases) {
    EXPECT_EQ(test_case.operand.ToString(), test_case.expected_string);
  }
}

TEST(InstructionOperandTest, AsTokenList) {
  const struct {
    InstructionOperand operand;
    std::vector<std::string> expected_tokens;
  } kTestCases[] = {
      {InstructionOperand::Register("RAX"), {"RAX"}},
      {InstructionOperand::ImmediateValue(123), {std::string(kImmediateToken)}},
      {InstructionOperand::FpImmediateValue(3.14),
       {std::string(kImmediateToken)}},
      {InstructionOperand::Address("RAX", 0, "", 0, ""),
       {std::string(kAddressToken), "RAX", std::string(kNoRegisterToken)}},
      {InstructionOperand::Address("RAX", -8, "RSI", 1, ""),
       {std::string(kAddressToken), "RAX", "RSI",
        std::string(kDisplacementToken)}},
      {InstructionOperand::MemoryLocation(1), {std::string(kMemoryToken)}},
  };
  for (const auto& test_case : kTestCases) {
    EXPECT_THAT(test_case.operand.AsTokenList(),
                ElementsAreArray(test_case.expected_tokens));
  }
}

TEST(AnnotationTest, Constructor) {
  constexpr char kName[] = "cache_miss_freq";
  constexpr double kValue = 0.875;

  const Annotation annotation(
      /* name = */ kName,
      /* value = */ kValue);
  EXPECT_EQ(annotation.name, kName);
  EXPECT_EQ(annotation.value, kValue);
}

TEST(AnnotationTest, ToString) {
  const Annotation annotation(
      /* name = */ "cache_miss_freq",
      /* value = */ 0.875);

  constexpr char kExpectedString[] =
      "Annotation(name='cache_miss_freq', value=0.875)";
  EXPECT_EQ(annotation.ToString(), kExpectedString);
}

TEST(InstructionTest, Constructor) {
  constexpr char kMnemonic[] = "MOV";
  constexpr char kLlvmMnemonic[] = "MOV32rr";
  const std::vector<std::string> kPrefixes = {"REP", "LOCK"};
  const std::vector<InstructionOperand> kInputOperands = {
      InstructionOperand::ImmediateValue(333)};
  const std::vector<InstructionOperand> kOutputOperands = {
      InstructionOperand::Register("RAX")};
  const std::vector<InstructionOperand> kImplicitInputOperands = {
      InstructionOperand::MemoryLocation(3)};
  const std::vector<InstructionOperand> kImplicitOutputOperands = {
      InstructionOperand::Register("EFLAGS")};
  const std::vector<Annotation> kInstructionAnnotations = {
      Annotation("cache_miss_freq", 0.875)};

  const Instruction instruction(
      /* mnemonic = */ kMnemonic,
      /* llvm_mnemonic = */ kLlvmMnemonic,
      /* prefixes = */ kPrefixes,
      /* input_operands = */ kInputOperands,
      /* implicit_input_operands = */ kImplicitInputOperands,
      /* output_operands = */ kOutputOperands,
      /* implicit_output_operands = */ kImplicitOutputOperands,
      /* instruction_annotations = */ kInstructionAnnotations);
  EXPECT_EQ(instruction.mnemonic, kMnemonic);
  EXPECT_EQ(instruction.llvm_mnemonic, kLlvmMnemonic);
  EXPECT_EQ(instruction.prefixes, kPrefixes);
  EXPECT_EQ(instruction.input_operands, kInputOperands);
  EXPECT_EQ(instruction.implicit_input_operands, kImplicitInputOperands);
  EXPECT_EQ(instruction.output_operands, kOutputOperands);
  EXPECT_EQ(instruction.implicit_output_operands, kImplicitOutputOperands);
  EXPECT_EQ(instruction.instruction_annotations, kInstructionAnnotations);
}

TEST(InstructionTest, AsTokenList) {
  constexpr char kMnemonic[] = "MOV";
  constexpr char kLlvmMnemonic[] = "MOV32rr";
  const std::vector<std::string> kPrefixes = {"REP", "LOCK"};
  const std::vector<InstructionOperand> kInputOperands = {
      InstructionOperand::ImmediateValue(333)};
  const std::vector<InstructionOperand> kOutputOperands = {
      InstructionOperand::Register("RAX")};
  const std::vector<InstructionOperand> kImplicitInputOperands = {
      InstructionOperand::MemoryLocation(3)};
  const std::vector<InstructionOperand> kImplicitOutputOperands = {
      InstructionOperand::Register("EFLAGS")};
  const std::vector<Annotation> kInstructionAnnotations = {
      Annotation("cache_miss_freq", 0.875)};

  const Instruction instruction(
      /* mnemonic = */ kMnemonic,
      /* llvm_mnemonic = */ kLlvmMnemonic,
      /* prefixes = */ kPrefixes,
      /* input_operands = */ kInputOperands,
      /* implicit_input_operands = */ kImplicitInputOperands,
      /* output_operands = */ kOutputOperands,
      /* implicit_output_operands = */ kImplicitOutputOperands,
      /* instruction_annotations = */ kInstructionAnnotations);

  EXPECT_THAT(instruction.AsTokenList(),
              ElementsAre(kPrefixes[0], kPrefixes[1], kMnemonic,
                          kDelimiterToken, "RAX", "EFLAGS", kDelimiterToken,
                          kImmediateToken, kMemoryToken, kDelimiterToken));
}

TEST(InstructionTest, ToString) {
  const Instruction instruction(
      /* mnemonic = */ "ADC",
      /* llvm_mnemonic = */ "ADC32rr",
      /* prefixes = */ {"LOCK"},
      /* input_operands = */
      {InstructionOperand::Register("RAX"),
       InstructionOperand::Register("RBX")},
      /* implicit_input_operands = */ {InstructionOperand::Register("EFLAGS")},
      /* output_operands = */ {InstructionOperand::Register("RAX")},
      /* implicit_output_operands = */
      {InstructionOperand::Register("EFLAGS")},
      /* instruction_annotations = */
      {Annotation("MEM_LOAD_RETIRED:L3_MISS", 0.875)});
  constexpr char kExpectedString[] =
      "Instruction(mnemonic='ADC', llvm_mnemonic='ADC32rr', "
      "prefixes=('LOCK',), "
      "input_operands=(InstructionOperand.from_register('RAX'), "
      "InstructionOperand.from_register('RBX'),), "
      "implicit_input_operands=(InstructionOperand.from_register('EFLAGS'),), "
      "output_operands=(InstructionOperand.from_register('RAX'),), "
      "implicit_output_operands=(InstructionOperand.from_register('EFLAGS'),), "
      "instruction_annotations=(Annotation(name='MEM_LOAD_RETIRED:L3_MISS', "
      "value=0.875),))";
  EXPECT_EQ(instruction.ToString(), kExpectedString);
}

TEST(InstructionTest, Equality) {
  Instruction instruction_1;
  Instruction instruction_2;

  EXPECT_EQ(instruction_1, instruction_2);
  EXPECT_FALSE(instruction_1 != instruction_2);

  // Set a different mnemonic on each.
  instruction_1.mnemonic = "MOV";
  EXPECT_NE(instruction_1, instruction_2);
  EXPECT_FALSE(instruction_1 == instruction_2);

  // Set the same mnemonic on both.
  instruction_2.mnemonic = "MOV";
  EXPECT_EQ(instruction_1, instruction_2);
  EXPECT_FALSE(instruction_1 != instruction_2);

  // Set a different LLVM mnemonic on each.
  instruction_1.llvm_mnemonic = "MOV32rr";
  EXPECT_NE(instruction_1, instruction_2);
  EXPECT_FALSE(instruction_1 == instruction_2);

  // Set the same LLVM mnemonic on both.
  instruction_2.llvm_mnemonic = "MOV32rr";
  EXPECT_EQ(instruction_1, instruction_2);
  EXPECT_FALSE(instruction_1 != instruction_2);

  // Set a different prefixes on each.
  instruction_1.prefixes = {"LOCK"};
  EXPECT_NE(instruction_1, instruction_2);
  EXPECT_FALSE(instruction_1 == instruction_2);

  // Set the same LLVM mnemonic on both.
  instruction_2.prefixes = {"LOCK"};
  EXPECT_EQ(instruction_1, instruction_2);
  EXPECT_FALSE(instruction_1 != instruction_2);

  // Set different input operands on each.
  instruction_1.input_operands = {InstructionOperand::MemoryLocation(12)};
  EXPECT_NE(instruction_1, instruction_2);
  EXPECT_FALSE(instruction_1 == instruction_2);

  // Set the same input operands on both.
  instruction_2.input_operands = {InstructionOperand::MemoryLocation(12)};
  EXPECT_EQ(instruction_1, instruction_2);
  EXPECT_FALSE(instruction_1 != instruction_2);

  // Set different implicit input operands on each.
  instruction_1.implicit_input_operands = {
      InstructionOperand::Register("EFLAGS")};
  EXPECT_NE(instruction_1, instruction_2);
  EXPECT_FALSE(instruction_1 == instruction_2);

  // Set the same implicit input operands on both.
  instruction_2.implicit_input_operands = {
      InstructionOperand::Register("EFLAGS")};
  EXPECT_EQ(instruction_1, instruction_2);
  EXPECT_FALSE(instruction_1 != instruction_2);

  // Set different output operands on each.
  instruction_1.output_operands = {InstructionOperand::Register("RDI")};
  EXPECT_NE(instruction_1, instruction_2);
  EXPECT_FALSE(instruction_1 == instruction_2);

  // Set the same output operands on both.
  instruction_2.output_operands = {InstructionOperand::Register("RDI")};
  EXPECT_EQ(instruction_1, instruction_2);
  EXPECT_FALSE(instruction_1 != instruction_2);

  // Set different implicit output operands on each.
  instruction_1.implicit_output_operands = {
      InstructionOperand::Register("EFLAGS")};
  EXPECT_NE(instruction_1, instruction_2);
  EXPECT_FALSE(instruction_1 == instruction_2);

  // Set the same implicit output operands on both.
  instruction_2.implicit_output_operands = {
      InstructionOperand::Register("EFLAGS")};
  EXPECT_EQ(instruction_1, instruction_2);
  EXPECT_FALSE(instruction_1 != instruction_2);
}

TEST(BasicBlockTest, Constructor) {
  const Instruction instruction(
      /* mnemonic = */ "ADC",
      /* llvm_mnemonic = */ "ADC32rr",
      /* prefixes = */ {"LOCK"},
      /* input_operands = */
      {InstructionOperand::Register("RAX"),
       InstructionOperand::Register("RBX")},
      /* implicit_input_operands = */ {InstructionOperand::Register("EFLAGS")},
      /* output_operands = */ {InstructionOperand::Register("RAX")},
      /* implicit_output_operands = */
      {InstructionOperand::Register("EFLAGS")},
      /* instruction_annotations = */
      {Annotation("MEM_LOAD_RETIRED:L3_MISS", 0.875)});

  const BasicBlock block({instruction});
  EXPECT_THAT(block.instructions, ElementsAre(instruction));
}

TEST(BasicBlockTest, Equality) {
  BasicBlock block_1;
  BasicBlock block_2;

  EXPECT_EQ(block_1, block_2);
  EXPECT_FALSE(block_1 != block_2);

  // Add an instruction to block_1.
  block_1.instructions.push_back(Instruction(
      /* mnemonic = */ "ADC",
      /* llvm_mnemonic = */ "ADC32rr",
      /* prefixes = */ {"LOCK"},
      /* input_operands = */
      {InstructionOperand::Register("RAX"),
       InstructionOperand::Register("RBX")},
      /* implicit_input_operands = */ {InstructionOperand::Register("EFLAGS")},
      /* output_operands = */ {InstructionOperand::Register("RAX")},
      /* implicit_output_operands = */
      {InstructionOperand::Register("EFLAGS")},
      /* instruction_annotations = */
      {Annotation("MEM_LOAD_RETIRED:L3_MISS", 0.875)}));

  EXPECT_NE(block_1, block_2);
  EXPECT_FALSE(block_1 == block_2);

  // Add the same instruction to block_2.
  block_2.instructions.push_back(Instruction(
      /* mnemonic = */ "ADC",
      /* llvm_mnemonic = */ "ADC32rr",
      /* prefixes = */ {"LOCK"},
      /* input_operands = */
      {InstructionOperand::Register("RAX"),
       InstructionOperand::Register("RBX")},
      /* implicit_input_operands = */ {InstructionOperand::Register("EFLAGS")},
      /* output_operands = */ {InstructionOperand::Register("RAX")},
      /* implicit_output_operands = */
      {InstructionOperand::Register("EFLAGS")},
      /* instruction_annotations = */
      {Annotation("MEM_LOAD_RETIRED:L3_MISS", 0.875)}));

  EXPECT_EQ(block_1, block_2);
  EXPECT_FALSE(block_1 != block_2);

  // Change the instruction in block_1.
  block_1.instructions.back().prefixes.clear();
  EXPECT_NE(block_1, block_2);
  EXPECT_FALSE(block_1 == block_2);
}

TEST(BasicBlockTest, ToString) {
  const Instruction instruction(
      /* mnemonic = */ "ADC",
      /* llvm_mnemonic = */ "ADC32rr",
      /* prefixes = */ {"LOCK"},
      /* input_operands = */
      {InstructionOperand::Register("RAX"),
       InstructionOperand::Register("RBX")},
      /* implicit_input_operands = */ {InstructionOperand::Register("EFLAGS")},
      /* output_operands = */ {InstructionOperand::Register("RAX")},
      /* implicit_output_operands = */
      {InstructionOperand::Register("EFLAGS")},
      /* instruction_annotations = */
      {Annotation("MEM_LOAD_RETIRED:L3_MISS", 0.875)});

  BasicBlock block({instruction});
  constexpr char kExpectedString[] =
      "BasicBlock(instructions=InstructionList((Instruction(mnemonic='ADC', "
      "llvm_mnemonic='ADC32rr', prefixes=('LOCK',), "
      "input_operands=(InstructionOperand.from_register('RAX'), "
      "InstructionOperand.from_register('RBX'),), "
      "implicit_input_operands=(InstructionOperand.from_register('EFLAGS'),), "
      "output_operands=(InstructionOperand.from_register('RAX'),), "
      "implicit_output_operands=(InstructionOperand.from_register('EFLAGS'),), "
      "instruction_annotations=(Annotation(name='MEM_LOAD_RETIRED:L3_MISS', "
      "value=0.875),)),"
      ")))";
  EXPECT_EQ(block.ToString(), kExpectedString);
}

}  // namespace
}  // namespace gematria
