// Copyright 2023 Google Inc.
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

#include "gematria/llvm/disassembler.h"

#include <cstdint>
#include <iterator>
#include <memory>

#include "absl/status/status.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/testing/llvm.h"
#include "gematria/testing/matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/include/llvm/MC/MCInst.h"
#include "llvm/include/llvm/MC/MCInstBuilder.h"
#include "llvm/include/llvm/MC/MCInstPrinter.h"
#include "llvm/lib/Target/X86/MCTargetDesc/X86MCTargetDesc.h"

namespace gematria {
namespace {

using ::testing::_;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::Matcher;
using ::testing::ResultOf;

template <typename AddressMatcher, typename AssemblyMatcher,
          typename MachineCodeMatcher, typename MCInstMatcherType>
testing::Matcher<DisassembledInstruction> IsDisassembledInstruction(
    AddressMatcher address, AssemblyMatcher assembly,
    MachineCodeMatcher machine_code, MCInstMatcherType mcinst) {
  return AllOf(Field("address", &DisassembledInstruction::address, address),
               Field("assembly", &DisassembledInstruction::assembly, assembly),
               Field("machine_code", &DisassembledInstruction::machine_code,
                     machine_code),
               Field("mc_inst", &DisassembledInstruction::mc_inst, mcinst));
}

Matcher<DisassembledInstruction> IsX86Nop(Matcher<uint64_t> address) {
  return IsDisassembledInstruction(
      address, EqualsNormalizingWhitespace("nop"), "\x90",
      IsMCInst(
          /*opcode_matcher=*/ResultOf("llvm::X86::isNOP", &llvm::X86::isNOP,
                                      testing::IsTrue()),
          /*operands_matcher=*/IsEmpty()));
}

Matcher<DisassembledInstruction> IsX86MovRaxRbx(Matcher<uint64_t> address) {
  return IsDisassembledInstruction(
      address, EqualsNormalizingWhitespace("mov rax, rbx"), "\x48\x89\xd8",
      IsMCInst(
          /*opcode_matcher=*/llvm::X86::MOV64rr,
          /*operands_matcher=*/ElementsAre(IsRegister(llvm::X86::RAX),
                                           IsRegister(llvm::X86::RBX))));
}

class DisassemblerTest : public testing::Test {
 protected:
  void SetUp() override { llvm_x86_64_ = LlvmArchitectureSupport::X86_64(); }

  std::unique_ptr<LlvmArchitectureSupport> llvm_x86_64_;
};

using AssembleInstructionTest = DisassemblerTest;

TEST_F(AssembleInstructionTest, X86_Nop) {
  std::unique_ptr<llvm::MCInstPrinter> printer =
      llvm_x86_64_->CreateMCInstPrinter(0);
  EXPECT_THAT(AssemblyFromMCInst(llvm_x86_64_->mc_instr_info(),
                                 llvm_x86_64_->mc_register_info(),
                                 llvm_x86_64_->mc_subtarget_info(), *printer,
                                 llvm::MCInstBuilder(llvm::X86::NOOP)),
              EqualsNormalizingWhitespace("nop"));
}

TEST_F(AssembleInstructionTest, X86_MovRaxRbx) {
  std::unique_ptr<llvm::MCInstPrinter> att_printer =
      llvm_x86_64_->CreateMCInstPrinter(0);
  std::unique_ptr<llvm::MCInstPrinter> intel_printer =
      llvm_x86_64_->CreateMCInstPrinter(1);
  EXPECT_THAT(
      AssemblyFromMCInst(llvm_x86_64_->mc_instr_info(),
                         llvm_x86_64_->mc_register_info(),
                         llvm_x86_64_->mc_subtarget_info(), *att_printer,
                         llvm::MCInstBuilder(llvm::X86::MOV64rr)
                             .addReg(llvm::X86::RAX)
                             .addReg(llvm::X86::RBX)),
      EqualsNormalizingWhitespace("movq %rbx, %rax"));
  EXPECT_THAT(
      AssemblyFromMCInst(llvm_x86_64_->mc_instr_info(),
                         llvm_x86_64_->mc_register_info(),
                         llvm_x86_64_->mc_subtarget_info(), *intel_printer,
                         llvm::MCInstBuilder(llvm::X86::MOV64rr)
                             .addReg(llvm::X86::RAX)
                             .addReg(llvm::X86::RBX)),
      EqualsNormalizingWhitespace("mov rax, rbx"));
}

using DisassembleOneInstructionTest = DisassemblerTest;

TEST_F(DisassembleOneInstructionTest, X86_Nop) {
  static constexpr uint8_t kInstructionData[] = {0x90};
  static constexpr uint64_t kAddress = 200;
  absl::Span<const uint8_t> instruction(kInstructionData);
  std::unique_ptr<llvm::MCInstPrinter> mc_inst_printer =
      llvm_x86_64_->CreateMCInstPrinter(1);

  EXPECT_THAT(
      DisassembleOneInstruction(
          llvm_x86_64_->mc_disassembler(), llvm_x86_64_->mc_instr_info(),
          llvm_x86_64_->mc_register_info(), llvm_x86_64_->mc_subtarget_info(),
          *mc_inst_printer, kAddress, instruction),
      IsOkAndHolds(IsX86Nop(kAddress)));
  EXPECT_THAT(instruction, IsEmpty());
}

TEST_F(DisassembleOneInstructionTest, X86_NopNopNop) {
  static constexpr uint8_t kInstructionData[] = {0x90, 0x90, 0x90};
  static constexpr uint64_t kAddress = 300;
  absl::Span<const uint8_t> instruction(kInstructionData);
  std::unique_ptr<llvm::MCInstPrinter> mc_inst_printer =
      llvm_x86_64_->CreateMCInstPrinter(1);

  EXPECT_THAT(
      DisassembleOneInstruction(
          llvm_x86_64_->mc_disassembler(), llvm_x86_64_->mc_instr_info(),
          llvm_x86_64_->mc_register_info(), llvm_x86_64_->mc_subtarget_info(),
          *mc_inst_printer, kAddress, instruction),
      IsOkAndHolds(IsX86Nop(kAddress)));
  EXPECT_THAT(instruction, ElementsAre(0x90, 0x90));
}

TEST_F(DisassembleOneInstructionTest, X86_MovRaxRbx) {
  static constexpr uint8_t kInstructionData[] = {0x48, 0x89, 0xd8};
  static constexpr uint64_t kAddress = 400;
  absl::Span<const uint8_t> instruction(kInstructionData);
  std::unique_ptr<llvm::MCInstPrinter> mc_inst_printer =
      llvm_x86_64_->CreateMCInstPrinter(1);

  EXPECT_THAT(
      DisassembleOneInstruction(
          llvm_x86_64_->mc_disassembler(), llvm_x86_64_->mc_instr_info(),
          llvm_x86_64_->mc_register_info(), llvm_x86_64_->mc_subtarget_info(),
          *mc_inst_printer, kAddress, instruction),
      IsOkAndHolds(IsX86MovRaxRbx(kAddress)));
  EXPECT_THAT(instruction, IsEmpty());
}

TEST_F(DisassembleOneInstructionTest, X86_EmptyInput) {
  static constexpr uint64_t kAddress = 500;
  absl::Span<const uint8_t> instruction;
  std::unique_ptr<llvm::MCInstPrinter> mc_inst_printer =
      llvm_x86_64_->CreateMCInstPrinter(1);

  EXPECT_THAT(
      DisassembleOneInstruction(
          llvm_x86_64_->mc_disassembler(), llvm_x86_64_->mc_instr_info(),
          llvm_x86_64_->mc_register_info(), llvm_x86_64_->mc_subtarget_info(),
          *mc_inst_printer, kAddress, instruction),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(DisassembleOneInstructionTest, X86_IncompleteMovRaxRbx) {
  static constexpr uint8_t kInstructionData[] = {0x48, 0x89};
  static constexpr uint64_t kAddress = 301;
  absl::Span<const uint8_t> instruction(kInstructionData);
  std::unique_ptr<llvm::MCInstPrinter> mc_inst_printer =
      llvm_x86_64_->CreateMCInstPrinter(1);

  EXPECT_THAT(
      DisassembleOneInstruction(
          llvm_x86_64_->mc_disassembler(), llvm_x86_64_->mc_instr_info(),
          llvm_x86_64_->mc_register_info(), llvm_x86_64_->mc_subtarget_info(),
          *mc_inst_printer, kAddress, instruction),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(instruction, ElementsAreArray(kInstructionData));
}

using DisassembleAllInstructionsTest = DisassemblerTest;

TEST_F(DisassembleAllInstructionsTest, NoInstructions) {
  static constexpr uint64_t kAddress = 302;
  absl::Span<const uint8_t> instruction;
  std::unique_ptr<llvm::MCInstPrinter> mc_inst_printer =
      llvm_x86_64_->CreateMCInstPrinter(1);

  EXPECT_THAT(
      DisassembleAllInstructions(
          llvm_x86_64_->mc_disassembler(), llvm_x86_64_->mc_instr_info(),
          llvm_x86_64_->mc_register_info(), llvm_x86_64_->mc_subtarget_info(),
          *mc_inst_printer, kAddress, instruction),
      IsOkAndHolds(IsEmpty()));
}

TEST_F(DisassembleAllInstructionsTest, X86_Nop) {
  static constexpr uint8_t kInstructionData[] = {0x90};
  static constexpr uint64_t kAddress = 302;
  std::unique_ptr<llvm::MCInstPrinter> mc_inst_printer =
      llvm_x86_64_->CreateMCInstPrinter(1);

  EXPECT_THAT(
      DisassembleAllInstructions(
          llvm_x86_64_->mc_disassembler(), llvm_x86_64_->mc_instr_info(),
          llvm_x86_64_->mc_register_info(), llvm_x86_64_->mc_subtarget_info(),
          *mc_inst_printer, kAddress, kInstructionData),
      IsOkAndHolds(ElementsAre(IsX86Nop(kAddress))));
}

TEST_F(DisassembleAllInstructionsTest, X86_NopNopNop) {
  static constexpr uint8_t kInstructionData[] = {0x90, 0x90, 0x90};
  static constexpr uint64_t kAddress = 303;
  static constexpr uint64_t kNopSize = 1;
  std::unique_ptr<llvm::MCInstPrinter> mc_inst_printer =
      llvm_x86_64_->CreateMCInstPrinter(1);

  EXPECT_THAT(
      DisassembleAllInstructions(
          llvm_x86_64_->mc_disassembler(), llvm_x86_64_->mc_instr_info(),
          llvm_x86_64_->mc_register_info(), llvm_x86_64_->mc_subtarget_info(),
          *mc_inst_printer, kAddress, kInstructionData),
      IsOkAndHolds(ElementsAre(IsX86Nop(kAddress),
                               IsX86Nop(kAddress + kNopSize),
                               IsX86Nop(kAddress + 2 * kNopSize))));
}

TEST_F(DisassembleOneInstructionTest, X86_NopMovRaxRbxNop) {
  static constexpr uint8_t kInstructionData[] = {0x90, 0x48, 0x89, 0xd8, 0x90};
  static constexpr uint64_t kAddress = 304;
  static constexpr uint64_t kNopSize = 1;
  static constexpr uint64_t kMov64RRSize = 3;
  std::unique_ptr<llvm::MCInstPrinter> mc_inst_printer =
      llvm_x86_64_->CreateMCInstPrinter(1);

  EXPECT_THAT(
      DisassembleAllInstructions(
          llvm_x86_64_->mc_disassembler(), llvm_x86_64_->mc_instr_info(),
          llvm_x86_64_->mc_register_info(), llvm_x86_64_->mc_subtarget_info(),
          *mc_inst_printer, kAddress, kInstructionData),
      IsOkAndHolds(ElementsAre(IsX86Nop(kAddress),
                               IsX86MovRaxRbx(kAddress + kNopSize),
                               IsX86Nop(kAddress + kNopSize + kMov64RRSize))));
}

TEST_F(DisassembleOneInstructionTest, X86_InvalidInstructionSequence) {
  // kInstructionData contains one `mov rax, rbx`, one `nop`, and then an
  // incomplete prefix of another `mov rax, rbx`.
  static constexpr uint8_t kInstructionData[] = {// mov rax, rbx
                                                 0x48, 0x89, 0xd8,
                                                 // nop
                                                 0x90,
                                                 // Incomplete mov rax, rbx
                                                 0x48, 0x89};
  static constexpr uint64_t kAddress = 304;
  absl::Span<const uint8_t> instruction(kInstructionData);
  std::unique_ptr<llvm::MCInstPrinter> mc_inst_printer =
      llvm_x86_64_->CreateMCInstPrinter(1);

  EXPECT_THAT(
      DisassembleAllInstructions(
          llvm_x86_64_->mc_disassembler(), llvm_x86_64_->mc_instr_info(),
          llvm_x86_64_->mc_register_info(), llvm_x86_64_->mc_subtarget_info(),
          *mc_inst_printer, kAddress, instruction),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace gematria
