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

#include "gematria/datasets/find_accessed_addrs_exegesis.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

// Use the absolute path for headers from llvm-exegesis as there is no
// canonical include path within LLVM as they are not properly exposed through
// a library and could potentially be confused with other LLVM includes.

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "gematria/llvm/asm_parser.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/proto/execution_annotation.pb.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/Error.h"
#include "llvm/tools/llvm-exegesis/lib/LlvmState.h"
#include "llvm/tools/llvm-exegesis/lib/TargetSelect.h"

using namespace llvm;
using namespace llvm::exegesis;

namespace gematria {
namespace {

class FindAccessedAddrsExegesisTest : public testing::Test {
 private:
  inline static std::unique_ptr<LlvmArchitectureSupport> LLVMArchSupport;
  LLVMState State;

 protected:
  FindAccessedAddrsExegesisTest()
      : State(cantFail(llvm::exegesis::LLVMState::Create("", "native"))) {}

  static void SetUpTestSuite() {
    LLVMArchSupport = LlvmArchitectureSupport::X86_64();
    InitializeX86ExegesisTarget();
  }

  std::string Assemble(std::string_view TextualAssembly) {
    auto MCInstsOrErr = gematria::ParseAsmCodeFromString(
        LLVMArchSupport->target_machine(), TextualAssembly,
        InlineAsm::AsmDialect::AD_ATT);
    CHECK_OK(MCInstsOrErr);
    const auto& MachineInstructions = *MCInstsOrErr;

    MCContext MachineCodeContext(
        LLVMArchSupport->target_machine().getTargetTriple(),
        &LLVMArchSupport->mc_asm_info(), &LLVMArchSupport->mc_register_info(),
        &LLVMArchSupport->mc_subtarget_info());
    const auto CodeEmitter =
        absl::WrapUnique(LLVMArchSupport->target().createMCCodeEmitter(
            LLVMArchSupport->mc_instr_info(), MachineCodeContext));

    SmallString<128> Code;
    SmallVector<MCFixup> Fixups;
    for (const auto& MachineInstruction : MachineInstructions)
      CodeEmitter->encodeInstruction(MachineInstruction, Code, Fixups,
                                     LLVMArchSupport->mc_subtarget_info());

    return std::string(Code);
  }

  llvm::Expected<ExecutionAnnotations> FindAccessedAddrsExegesis(
      std::string_view TextualAssembly) {
    auto Code = Assemble(TextualAssembly);
    auto Annotator = cantFail(ExegesisAnnotator::create(State));
    return Annotator->findAccessedAddrs(
        llvm::ArrayRef(reinterpret_cast<const uint8_t*>(Code.data()),
                       Code.size()),
        50);
  }
};

// TODO(boomanaiden154): The boolean static casting to check the status of the
// Expecteds in the following two test cases can be replaced with
// ASSERT_THAT_EXPECTED once pthread errors that occur when linking
// llvm:TestingSupport are fixed.
TEST_F(FindAccessedAddrsExegesisTest, ExegesisNoAccess) {
  auto AddrsOrErr = FindAccessedAddrsExegesis(R"asm(
    movq %r11, %r12
  )asm");
  ASSERT_TRUE(static_cast<bool>(AddrsOrErr));
  ExecutionAnnotations Result = *AddrsOrErr;
  EXPECT_EQ(Result.accessed_blocks_size(), 0);
}

TEST_F(FindAccessedAddrsExegesisTest, ExegesisOneAccess) {
  auto AddrsOrErr = FindAccessedAddrsExegesis(R"asm(
    movq $0x10000, %rax
    movq (%rax), %rax
  )asm");
  ASSERT_TRUE(static_cast<bool>(AddrsOrErr));
  ExecutionAnnotations Result = *AddrsOrErr;
  EXPECT_EQ(Result.accessed_blocks_size(), 1);
  EXPECT_EQ(Result.accessed_blocks()[0], 0x10000);
}

TEST_F(FindAccessedAddrsExegesisTest, ExegesisNotPageAligned) {
  auto AddrsOrErr = FindAccessedAddrsExegesis(R"asm(
    movq $0x10001, %rax
    movq (%rax), %rax
  )asm");
  ASSERT_TRUE(static_cast<bool>(AddrsOrErr));
  ExecutionAnnotations Result = *AddrsOrErr;
  EXPECT_EQ(Result.accessed_blocks_size(), 1);
  EXPECT_EQ(Result.accessed_blocks()[0], 0x10000);
}

TEST_F(FindAccessedAddrsExegesisTest, ExegesisZeroAddressError) {
  auto AddrsOrErr = FindAccessedAddrsExegesis(R"asm(
    movq $0x0, %rax
    movq (%rax), %rax
  )asm");
  ASSERT_FALSE(static_cast<bool>(AddrsOrErr));
}

TEST_F(FindAccessedAddrsExegesisTest, ExegesisMultipleSameAddressError) {
  // Try and load memory from an address above the current user space address
  // space ceiling (assuming five level page tables are not enabled) as the
  // script will currently try and annotate this, but exegesis will fail
  // to map the address when it attempts to.
  auto AddrsOrErr = FindAccessedAddrsExegesis(R"asm(
    movabsq $0x0000800000000000, %rax
    movq (%rax), %rax
  )asm");
  ASSERT_FALSE(static_cast<bool>(AddrsOrErr));
}

// This test is disabled due to taking ~20 seconds to run.
// TODO(boomanaiden154): Make this test run as part of an "expensive checks"
// configuration.
TEST_F(FindAccessedAddrsExegesisTest, DISABLED_QuitMaxAnnotationAttempts) {
  auto AddrsOrErr = FindAccessedAddrsExegesis(R"asm(
    movq (%rax), %rdx
    addq $0x1000, %rax
  )asm");
  ASSERT_FALSE(static_cast<bool>(AddrsOrErr));
}

}  // namespace
}  // namespace gematria
