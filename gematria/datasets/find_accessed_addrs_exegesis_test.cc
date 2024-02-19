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

// Use the absolute path for headers from llvm-exegesis as there is no
// canonical include path within LLVM as they are not properly exposed through
// a library and could potentially be confused with other LLVM includes.

#include "absl/log/check.h"
#include "gematria/llvm/asm_parser.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gtest/gtest.h"
#include "llvm/MC/MCCodeEmitter.h"
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

  llvm::Expected<AccessedAddrs> FindAccessedAddrsExegesis(
      std::string_view TextualAssembly) {
    auto Code = Assemble(TextualAssembly);
    auto Annotator = cantFail(ExegesisAnnotator::create(State));
    return Annotator->findAccessedAddrs(llvm::ArrayRef(
        reinterpret_cast<const uint8_t*>(Code.data()), Code.size()));
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
  AccessedAddrs Result = *AddrsOrErr;
  EXPECT_EQ(Result.accessed_blocks.size(), 0);
}

TEST_F(FindAccessedAddrsExegesisTest, ExegesisOneAccess) {
  auto AddrsOrErr = FindAccessedAddrsExegesis(R"asm(
    movq $0x10000, %rax
    movq (%rax), %rax
  )asm");
  ASSERT_TRUE(static_cast<bool>(AddrsOrErr));
  AccessedAddrs Result = *AddrsOrErr;
  EXPECT_EQ(Result.accessed_blocks.size(), 1);
  EXPECT_EQ(Result.accessed_blocks[0], 0x10000);
}

TEST_F(FindAccessedAddrsExegesisTest, ExegesisNotPageAligned) {
  auto AddrsOrErr = FindAccessedAddrsExegesis(R"asm(
    movq $0x10001, %rax
    movq (%rax), %rax
  )asm");
  ASSERT_TRUE(static_cast<bool>(AddrsOrErr));
  AccessedAddrs Result = *AddrsOrErr;
  EXPECT_EQ(Result.accessed_blocks.size(), 1);
  EXPECT_EQ(Result.accessed_blocks[0], 0x10000);
}

}  // namespace
}  // namespace gematria
