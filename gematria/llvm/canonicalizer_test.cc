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

#include "gematria/llvm/canonicalizer.h"

#include <memory>
#include <string>
#include <vector>

#include "gematria/basic_block/basic_block.h"
#include "gematria/llvm/asm_parser.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"

namespace gematria {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Not;

class X86BasicBlockExtractorTest : public testing::Test {
 protected:
  void SetUp() override {
    llvm_architecture_ = LlvmArchitectureSupport::X86_64();

    extractor_ = std::make_unique<const X86Canonicalizer>(
        &llvm_architecture_->target_machine());
  }

  std::vector<llvm::MCInst> ParseAssemblyCode(std::string assembly) const {
    return ParseAsmCodeFromString(llvm_architecture_->target_machine(),
                                  assembly, llvm::InlineAsm::AD_Intel)
        .value();
  }

  std::unique_ptr<const LlvmArchitectureSupport> llvm_architecture_;
  std::unique_ptr<llvm::MCInstPrinter> inst_printer_;
  std::unique_ptr<const X86Canonicalizer> extractor_;
};

TEST_F(X86BasicBlockExtractorTest, InstructionFromMCInst) {
  const std::vector<llvm::MCInst> mcinsts = ParseAssemblyCode(R"(
      ADD RAX, RBX
  )");
  ASSERT_THAT(mcinsts, Not(IsEmpty()));

  EXPECT_EQ(extractor_->InstructionFromMCInst(mcinsts[0]),
            Instruction(
                /* mnemonic= */ "ADD", /* llvm_mnemonic= */ "ADD64rr",
                /* prefixes= */ {},
                /* input_operands= */
                {InstructionOperand::Register("RAX"),
                 InstructionOperand::Register("RBX")},
                /* implicit_input_operands= */ {},
                /* output_operands= */ {InstructionOperand::Register("RAX")},
                /* implicit_output_operands= */
                {InstructionOperand::Register("EFLAGS")}));
}

TEST_F(X86BasicBlockExtractorTest, BasicBlockFromMCInst) {
  const std::vector<llvm::MCInst> mcinsts = ParseAssemblyCode(R"(
      ADD RAX, RBX
      XOR QWORD PTR[RCX], RAX
  )");
  ASSERT_THAT(mcinsts, Not(IsEmpty()));

  const BasicBlock block = extractor_->BasicBlockFromMCInst(mcinsts);
  EXPECT_THAT(
      block.instructions,
      ElementsAre(
          Instruction(
              /* mnemonic= */ "ADD", /* llvm_mnemonic= */ "ADD64rr",
              /* prefixes= */ {},
              /* input_operands= */
              {InstructionOperand::Register("RAX"),
               InstructionOperand::Register("RBX")},
              /* implicit_input_operands= */ {},
              /* output_operands= */ {InstructionOperand::Register("RAX")},
              /* implicit_output_operands= */
              {InstructionOperand::Register("EFLAGS")}),
          Instruction(
              /* mnemonic= */ "XOR", /* llvm_mnemonic= */ "XOR64mr",
              /* prefixes= */ {},
              /* input_operands= */
              {InstructionOperand::MemoryLocation(1),
               InstructionOperand::Address(
                   /* base_register= */ "RCX",
                   /* displacement= */ 0,
                   /* index_register= */ std::string(),
                   /* scaling= */ 1,
                   /* segment_register= */ std::string()),
               InstructionOperand::Register("RAX")},
              /* implicit_input_operands= */ {},
              /* output_operands= */ {InstructionOperand::MemoryLocation(1)},
              /* implicit_output_operands= */
              {InstructionOperand::Register("EFLAGS")})));
}

TEST_F(X86BasicBlockExtractorTest, InstructionWithExprOperand) {
  const std::vector<llvm::MCInst> mcinsts = ParseAssemblyCode(R"(
    loop:
      CMP RAX, RBX
      JE loop + 10
  )");
  ASSERT_THAT(mcinsts, Not(IsEmpty()));

  const BasicBlock block = extractor_->BasicBlockFromMCInst(mcinsts);
  EXPECT_THAT(
      block.instructions,
      ElementsAre(
          Instruction(/* mnemonic= */ "CMP", /* llvm_mnemonic= */ "CMP64rr",
                      /* prefixes= */ {},
                      /* input_operands= */
                      {InstructionOperand::Register("RAX"),
                       InstructionOperand::Register("RBX")},
                      /* implicit_input_operands = */ {},
                      /* output_operands= */ {},
                      /* implicit_output_operands= */
                      {InstructionOperand::Register("EFLAGS")}),
          Instruction(/* mnemonic= */ "JE", /* llvm_mnemonic= */ "JCC_1",
                      /* prefixes= */ {},
                      /* input_operands= */
                      {InstructionOperand::ImmediateValue(1),
                       InstructionOperand::ImmediateValue(4)},
                      /* implicit_input_operands= */
                      {InstructionOperand::Register("EFLAGS")},
                      /* output_operands= */ {},
                      /* implicit_output_operands= */ {})));
}

TEST_F(X86BasicBlockExtractorTest, InstructionWithSegmentRegisters) {
  const std::vector<llvm::MCInst> mcinsts = ParseAssemblyCode(R"(
      MOV RAX, QWORD PTR FS:[RSI + 123]
  )");
  ASSERT_THAT(mcinsts, Not(IsEmpty()));

  EXPECT_EQ(
      extractor_->InstructionFromMCInst(mcinsts[0]),
      Instruction(
          /* mnemonic= */ "MOV", /* llvm_mnemonic= */ "MOV64rm",
          /* prefixes= */ {},
          /* input_operands= */
          {InstructionOperand::MemoryLocation(1),
           InstructionOperand::Address(/* base_register= */ "RSI",
                                       /* displacement= */ 123,
                                       /* index_register= */ std::string(),
                                       /* scaling= */ 1,
                                       /* segment_register= */ "FS")},
          /* implicit_input_operands= */ {},
          /* output_operands= */ {InstructionOperand::Register("RAX")},
          /* implicit_output_operands= */ {}));
}

TEST_F(X86BasicBlockExtractorTest, LonePrefix) {
  const std::vector<llvm::MCInst> mcinsts = ParseAssemblyCode(R"(
      LOCK
  )");
  ASSERT_THAT(mcinsts, Not(IsEmpty()));

  EXPECT_EQ(extractor_->InstructionFromMCInst(mcinsts[0]),
            Instruction(
                /* mnemonic= */ "lock", /* llvm_mnemonic= */ "LOCK_PREFIX",
                /* prefixes= */ {},
                /* input_operands= */ {},
                /* implicit_input_operands= */ {},
                /* output_operands= */ {},
                /* implicit_output_operands= */ {}));
}

TEST_F(X86BasicBlockExtractorTest, RepMov) {
  const std::vector<llvm::MCInst> mcinsts = ParseAssemblyCode(R"(
    rep mov eax, 1
  )");
  ASSERT_THAT(mcinsts, Not(IsEmpty()));

  EXPECT_EQ(extractor_->InstructionFromMCInst(mcinsts[0]),
            Instruction("MOV", "MOV32ri", {"REP"},
                        {InstructionOperand::ImmediateValue(1)}, {},
                        {InstructionOperand::Register("EAX")}, {}));
}

}  // namespace
}  // namespace gematria
