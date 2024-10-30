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

#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/log/check.h"
#include "gematria/llvm/asm_parser.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/MC/MCInst.h"
#include "llvm/lib/Target/X86/MCTargetDesc/X86MCTargetDesc.h"

using testing::AnyOf;
using testing::IsEmpty;
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

  std::vector<unsigned> getUsedRegs(std::string_view TextualAssembly) {
    return getUsedRegisters(getInstructions(TextualAssembly),
                            LlvmArchSupport->mc_register_info(),
                            LlvmArchSupport->mc_instr_info());
  }

  std::optional<unsigned> getUnusedGPReg(std::string_view TextualAssembly) {
    return getUnusedGPRegister(getInstructions(TextualAssembly),
                               LlvmArchSupport->mc_register_info(),
                               LlvmArchSupport->mc_instr_info());
  }
};

TEST_F(BasicBlockUtilsTest, UsedRegistersSingleRegister) {
  std::vector<unsigned> UsedRegisters = getUsedRegs(R"asm(
    mov %rax, %rcx
  )asm");
  EXPECT_THAT(UsedRegisters, UnorderedElementsAre(X86::RAX));
}

TEST_F(BasicBlockUtilsTest, UsedRegistersSubRegister) {
  std::vector<unsigned> UsedRegisters = getUsedRegs(R"asm(
    mov %al, %cl
    mov %ax, %cx
    mov %rax, %rcx
  )asm");
  EXPECT_THAT(UsedRegisters, UnorderedElementsAre(X86::RAX));
}

TEST_F(BasicBlockUtilsTest, UsedRegistersMultipleRegisters) {
  std::vector<unsigned> UsedRegisters = getUsedRegs(R"asm(
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
  std::vector<unsigned> UsedRegisters = getUsedRegs(R"asm(
    vmovapd %zmm1, %zmm2
  )asm");
  EXPECT_THAT(UsedRegisters, UnorderedElementsAre(X86::ZMM1));
}

TEST_F(BasicBlockUtilsTest, UsedRegistersVectorSubRegisters) {
  std::vector<unsigned> UsedRegisters = getUsedRegs(R"asm(
    movaps %xmm1, %xmm2
  )asm");
  EXPECT_THAT(UsedRegisters, UnorderedElementsAre(X86::XMM1));
}

TEST_F(BasicBlockUtilsTest, UsedRegistersImplicitUse) {
  std::vector<unsigned> UsedRegisters = getUsedRegs(R"asm(
    pushq %rax
  )asm");
  EXPECT_THAT(UsedRegisters, UnorderedElementsAre(X86::RAX, X86::RSP));
}

TEST_F(BasicBlockUtilsTest, UsedRegistersImplicitDefs) {
  std::vector<unsigned> UsedRegisters = getUsedRegs(R"asm(
    rdtsc
    movq %rax, %r8
    movq %rdx, %r8
  )asm");
  EXPECT_THAT(UsedRegisters, IsEmpty());
}

TEST_F(BasicBlockUtilsTest, UsedRegistersCPUID) {
  std::vector<unsigned> UsedRegisters = getUsedRegs(R"asm(
    cpuid
    movq %rbx, %r8
    movq %rdx, %r8
  )asm");
  EXPECT_THAT(UsedRegisters, UnorderedElementsAre(X86::RAX, X86::RCX));
}

TEST_F(BasicBlockUtilsTest, UsedRegistersUseDefSameRegister) {
  std::vector<unsigned> UsedRegisters = getUsedRegs(R"asm(
    addq %rax, %rbx
  )asm");
  EXPECT_THAT(UsedRegisters, UnorderedElementsAre(X86::RAX, X86::RBX));
}

TEST_F(BasicBlockUtilsTest, UsedRegistersRegisterAliasing) {
  std::vector<unsigned> UsedRegisters = getUsedRegs(R"asm(
    movb 1, %al
    addq %rax, %rbx
  )asm");
  EXPECT_THAT(UsedRegisters, UnorderedElementsAre(X86::RAX, X86::RBX));
}

TEST_F(BasicBlockUtilsTest, UsedRegistersRegisterAliasing32Bit) {
  std::vector<unsigned> UsedRegisters = getUsedRegs(R"asm(
    movl $1, %eax
    add %rax, %rbx
  )asm");
  EXPECT_THAT(UsedRegisters, UnorderedElementsAre(X86::RBX));
}

TEST_F(BasicBlockUtilsTest, UsedRegistersAddressingModes) {
  std::vector<unsigned> UsedRegisters = getUsedRegs(R"asm(
    add %rax, -16(%rbx, %rcx, 8)
  )asm");
  EXPECT_THAT(UsedRegisters,
              UnorderedElementsAre(X86::RAX, X86::RBX, X86::RCX));
}

// TODO(boomanaiden154): Currently, this returns %RCX in addition to %RAX and
// %RDX, when it should only return the latter two. This is not a large concern
// as we are still returning a safe register set, but this should be fixed
// eventually.
TEST_F(BasicBlockUtilsTest, DISABLED_UsedRegistersSingleByteDefines) {
  std::vector<unsigned> UsedRegisters = getUsedRegs(R"asm(
    movb %al, %cl
    movb %cl, %dl
    addq %rdx, %rax
  )asm");
  EXPECT_THAT(UsedRegisters,
              UnorderedElementsAre(X86::RAX, X86::RDX, X86::RCX));
}

TEST_F(BasicBlockUtilsTest, MovsqImplicitDfUsesEflags) {
  std::vector<unsigned> UsedRegisters = getUsedRegs(R"asm(
    movsq
  )asm");
  EXPECT_THAT(UsedRegisters,
              UnorderedElementsAre(X86::RSI, X86::RDI, X86::EFLAGS));
}

TEST_F(BasicBlockUtilsTest, UnusedGPRegisterSingleInstruction) {
  std::optional<unsigned> UnusedGPRegister = getUnusedGPReg(R"asm(
    mov %rax, %rcx
  )asm");
  EXPECT_THAT(*UnusedGPRegister,
              AnyOf(X86::RDX, X86::RBX, X86::RSI, X86::RDI, X86::RSP, X86::RBP,
                    X86::R8, X86::R9, X86::R10, X86::R11, X86::R12, X86::R13,
                    X86::R14, X86::R15));
}

TEST_F(BasicBlockUtilsTest, UnusedGPRegisterImplicitUseDef) {
  std::optional<unsigned> UnusedGPRegister = getUnusedGPReg(R"asm(
    pushq %rax
    pushq %rcx
    pushq %rdx
    pushq %rbx
    pushq %rsi
    pushq %rdi
  )asm");
  EXPECT_THAT(*UnusedGPRegister, AnyOf(X86::R8, X86::R9, X86::R10, X86::R11,
                                       X86::R12, X86::R13, X86::R14, X86::R15));
}

TEST_F(BasicBlockUtilsTest, UnusedGPRegisterFullPressure) {
  std::optional<unsigned> UnusedGPRegister = getUnusedGPReg(R"asm(
    movq %rax, %rcx
    movq %rdx, %rbx
    movq %rsi, %rdi
    movq %rsp, %rbp
    movq %r8, %r9
    movq %r10, %r11
    movq %r12, %r13
    movq %r14, %r15
  )asm");
  EXPECT_FALSE(UnusedGPRegister.has_value());
}

}  // namespace
}  // namespace gematria
