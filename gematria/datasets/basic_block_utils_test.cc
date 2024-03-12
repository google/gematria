// Copyright 2024 Google Inc.
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

#include "gematria/datasets/basic_block_utils.h"

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86RegisterInfo.h"
#include "gematria/llvm/asm_parser.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/testing/matchers.h"
#include "gtest/gtest.h"

using testing::UnorderedElementsAre;

namespace gematria {
namespace {
class BasicBlockUtilsTest : public ::testing::Test {
 private:
  inline static std::unique_ptr<LlvmArchitectureSupport> LlvmArchSupport;

 protected:
  static void SetUpTestSuite() {
    LlvmArchSupport = LlvmArchitectureSupport::X86_64();
  }

  std::vector<DisassembledInstruction> getInstructions(
      std::string_view TextualAssembly) {
    auto MCInstsOrError = gematria::ParseAsmCodeFromString(
        LlvmArchSupport->target_machine(), TextualAssembly,
        llvm::InlineAsm::AsmDialect::AD_ATT);
    CHECK_OK(MCInstsOrError);

    std::vector<DisassembledInstruction> Instructions;
    Instructions.reserve(MCInstsOrError->size());

    for (MCInst Instruction : *MCInstsOrError)
      Instructions.push_back({0, "", "", Instruction});

    return Instructions;
  }

  std::vector<unsigned> getUsedRegisters(std::string_view TextualAssembly) {
    return BasicBlockUtils::getUsedRegisters(
        getInstructions(TextualAssembly), LlvmArchSupport->mc_register_info(),
        LlvmArchSupport->mc_instr_info());
  }

  std::optional<unsigned> getLoopRegister(std::string_view TextualAssembly) {
    return BasicBlockUtils::getLoopRegister(getInstructions(TextualAssembly),
                                            LlvmArchSupport->mc_register_info(),
                                            LlvmArchSupport->mc_instr_info());
  }
};

TEST_F(BasicBlockUtilsTest, UsedRegistersSingleRegister) {
  std::vector<unsigned> UsedRegisters = getUsedRegisters(R"asm(
    mov %rax, %rcx
  )asm");
  EXPECT_THAT(UsedRegisters, UnorderedElementsAre(X86::RAX));
}

TEST_F(BasicBlockUtilsTest, UsedRegistersSubRegister) {
  std::vector<unsigned> UsedRegisters = getUsedRegisters(R"asm(
    mov %al, %cl
    mov %ax, %cx
    mov %rax, %rcx
  )asm");
  EXPECT_THAT(UsedRegisters, UnorderedElementsAre(X86::RAX));
}

TEST_F(BasicBlockUtilsTest, UsedRegistersMultipleRegisters) {
  std::vector<unsigned> UsedRegisters = getUsedRegisters(R"asm(
    movq %rax, %rcx
    movq %rdx, %rbx
    movq %rsi, %rdi
    movq %rsp, %rbp
    movq %r8, %r9
    movq %r10, %r11
    movq %r12, %r13
    movq %r14, %r15
  )asm");
  EXPECT_THAT(UsedRegisters,
              UnorderedElementsAre(X86::RAX, X86::RDX, X86::RSI, X86::RSP,
                                   X86::R8, X86::R10, X86::R12, X86::R14));
}

TEST_F(BasicBlockUtilsTest, UsedRegistersVectorRegisters) {
  std::vector<unsigned> UsedRegisters = getUsedRegisters(R"asm(
    vmovapd %zmm1, %zmm2
  )asm");
  EXPECT_THAT(UsedRegisters, UnorderedElementsAre(X86::ZMM1));
}

TEST_F(BasicBlockUtilsTest, UsedRegistersVectorSubRegisters) {
  std::vector<unsigned> UsedRegisters = getUsedRegisters(R"asm(
    movaps %xmm1, %xmm2
  )asm");
  EXPECT_THAT(UsedRegisters, UnorderedElementsAre(X86::XMM1));
}

TEST_F(BasicBlockUtilsTest, UsedRegistersImplicitUse) {
  std::vector<unsigned> UsedRegisters = getUsedRegisters(R"asm(
    pushq %rax
  )asm");
  EXPECT_THAT(UsedRegisters, UnorderedElementsAre(X86::RAX, X86::RSP));
}

TEST_F(BasicBlockUtilsTest, LoopRegisterSingleInstruction) {
  std::optional<unsigned> LoopRegister = getLoopRegister(R"asm(
    mov %rax, %rcx
  )asm");
  EXPECT_EQ(*LoopRegister, X86::RDX);
}

TEST_F(BasicBlockUtilsTest, LoopRegisterImplicitUseDef) {
  std::optional<unsigned> LoopRegister = getLoopRegister(R"asm(
    pushq %rax
    pushq %rcx
    pushq %rdx
    pushq %rbx
    pushq %rsi
    pushq %rdi
  )asm");
  EXPECT_EQ(*LoopRegister, X86::R8);
}

TEST_F(BasicBlockUtilsTest, LoopRegisterFullPressure) {
  std::optional<unsigned> LoopRegister = getLoopRegister(R"asm(
    movq %rax, %rcx
    movq %rdx, %rbx
    movq %rsi, %rdi
    movq %rsp, %rbp
    movq %r8, %r9
    movq %r10, %r11
    movq %r12, %r13
    movq %r14, %r15
  )asm");
  EXPECT_FALSE(LoopRegister.has_value());
}

}  // namespace
}  // namespace gematria
