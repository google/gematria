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
#include "gmock/gmock.h"
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

using ::testing::Contains;
using ::testing::IsSupersetOf;
using ::testing::Property;

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

TEST_F(FindAccessedAddrsExegesisTest, DFRegister) {
  // Test that we can successfully find the accessed addrs for a movsq
  // instruction, which makes things more complicated by explicitly using
  // the df register. We do not care about the specific addresses in this
  // case.
  auto AddrsOrErr = FindAccessedAddrsExegesis(R"asm(
    movsq
  )asm");
  ASSERT_TRUE(static_cast<bool>(AddrsOrErr));
}

// Test that we can annotate snippets using various different register
// classes. This is mostly intended to test that we do not get any MC
// verification errors, which exegesis will throw if we do not define a
// register that gets used in the snippet.

TEST_F(FindAccessedAddrsExegesisTest, MXCSRRegister) {
  auto AddrsOrErr = FindAccessedAddrsExegesis(R"asm(
    stmxcsr (%rax)
  )asm");
  ASSERT_TRUE(static_cast<bool>(AddrsOrErr));
  EXPECT_THAT(AddrsOrErr->initial_registers(),
              Contains(Property("register_name",
                                &RegisterAndValue::register_name, "MXCSR")));
}

TEST_F(FindAccessedAddrsExegesisTest, MMXRegisters) {
  auto AddrsOrErr = FindAccessedAddrsExegesis(R"asm(
    movq %mm0, (%rax)
  )asm");
  ASSERT_TRUE(static_cast<bool>(AddrsOrErr));
  EXPECT_THAT(AddrsOrErr->initial_registers(),
              Contains(Property("register_name",
                                &RegisterAndValue::register_name, "MM0")));
}

TEST_F(FindAccessedAddrsExegesisTest, XMMRegisters) {
  auto AddrsOrErr = FindAccessedAddrsExegesis(R"asm(
    vmovdqu %xmm0, (%rax)
  )asm");
  ASSERT_TRUE(static_cast<bool>(AddrsOrErr));
  EXPECT_THAT(AddrsOrErr->initial_registers(),
              Contains(Property("register_name",
                                &RegisterAndValue::register_name, "XMM0")));
}

TEST_F(FindAccessedAddrsExegesisTest, YMMRegisters) {
  auto AddrsOrErr = FindAccessedAddrsExegesis(R"asm(
    vmovdqu %ymm0, (%rax)
  )asm");
  ASSERT_TRUE(static_cast<bool>(AddrsOrErr));
  EXPECT_THAT(AddrsOrErr->initial_registers(),
              Contains(Property("register_name",
                                &RegisterAndValue::register_name, "YMM0")));
}

TEST_F(FindAccessedAddrsExegesisTest, AVX512KZMMRegisters) {
  // This test requires AVX512, skip if we are not running on a CPU with
  // AVX512F.
  if (!__builtin_cpu_supports("avx512f")) {
    GTEST_SKIP() << "CPU does not support AVX512, skipping.";
  }

  auto AddrsOrErr = FindAccessedAddrsExegesis(R"asm(
    vmovdqu32 %zmm0, (%rax) {%k1}
  )asm");
  ASSERT_TRUE(static_cast<bool>(AddrsOrErr));
  EXPECT_THAT(
      AddrsOrErr->initial_registers(),
      IsSupersetOf(
          {Property("register_name", &RegisterAndValue::register_name, "ZMM0"),
           Property("register_name", &RegisterAndValue::register_name, "K1")}));
}

TEST_F(FindAccessedAddrsExegesisTest, SegmentRegisters) {
  auto AddrsOrErr = FindAccessedAddrsExegesis(R"asm(
    movq %fs:4, %rax
    movq %gs:4, %rdx
  )asm");
  ASSERT_TRUE(static_cast<bool>(AddrsOrErr));
  EXPECT_THAT(
      AddrsOrErr->initial_registers(),
      IsSupersetOf(
          {Property("register_name", &RegisterAndValue::register_name, "FS"),
           Property("register_name", &RegisterAndValue::register_name, "GS")}));
}

TEST_F(FindAccessedAddrsExegesisTest, FSTCWRegister) {
  auto AddrsOrErr = FindAccessedAddrsExegesis(R"asm(
    fstcw (%rax)
  )asm");
  ASSERT_TRUE(static_cast<bool>(AddrsOrErr));
  EXPECT_THAT(AddrsOrErr->initial_registers(),
              Contains(Property("register_name",
                                &RegisterAndValue::register_name, "FPCW")));
}

TEST_F(FindAccessedAddrsExegesisTest, STRegister) {
  auto AddrsOrErr = FindAccessedAddrsExegesis(R"asm(
    fadd
  )asm");
  ASSERT_TRUE(static_cast<bool>(AddrsOrErr));
  EXPECT_THAT(AddrsOrErr->initial_registers(),
              Contains(Property("register_name",
                                &RegisterAndValue::register_name, "ST1")));
}

}  // namespace
}  // namespace gematria
