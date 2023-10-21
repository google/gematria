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

#include "gematria/llvm/asm_parser.h"

#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>

#include "absl/status/status.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/testing/llvm.h"
#include "gematria/testing/matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/MC/MCInst.h"
#include "lib/Target/X86/MCTargetDesc/X86BaseInfo.h"  //// IWYU pragma: keep (for opcodes).
#include "lib/Target/X86/MCTargetDesc/X86MCTargetDesc.h"

namespace gematria {
namespace {

using ::testing::ElementsAre;

class AsmParserTest : public testing::Test {
 protected:
  void SetUp() override { llvm_x86_ = LlvmArchitectureSupport::X86_64(); }

  std::unique_ptr<LlvmArchitectureSupport> llvm_x86_;
};

auto HasOpcode(int opcode) {
  return testing::Property(&llvm::MCInst::getOpcode, opcode);
}

TEST_F(AsmParserTest, IntelAssembly) {
  static constexpr std::string_view kAssembly = R"asm(
    xor eax, eax
    add eax, 1
  )asm";
  const auto result = ParseAsmCodeFromString(
      llvm_x86_->target_machine(), kAssembly, llvm::InlineAsm::AD_Intel);
  EXPECT_THAT(
      result,
      IsOkAndHolds(ElementsAre(
          IsMCInst(llvm::X86::XOR32rr, ElementsAre(IsRegister(llvm::X86::EAX),
                                                   IsRegister(llvm::X86::EAX),
                                                   IsRegister(llvm::X86::EAX))),
          IsMCInst(llvm::X86::ADD32ri8,
                   ElementsAre(IsRegister(llvm::X86::EAX),
                               IsRegister(llvm::X86::EAX), IsImmediate(1))))));
}

TEST_F(AsmParserTest, ATTAssembly) {
  static constexpr std::string_view kAssembly = R"asm(
    xorl %eax, %eax
    addl $1, %eax
  )asm";
  const auto result = ParseAsmCodeFromString(
      llvm_x86_->target_machine(), kAssembly, llvm::InlineAsm::AD_ATT);
  EXPECT_THAT(
      result,
      IsOkAndHolds(ElementsAre(
          IsMCInst(llvm::X86::XOR32rr, ElementsAre(IsRegister(llvm::X86::EAX),
                                                   IsRegister(llvm::X86::EAX),
                                                   IsRegister(llvm::X86::EAX))),
          IsMCInst(llvm::X86::ADD32ri8,
                   ElementsAre(IsRegister(llvm::X86::EAX),
                               IsRegister(llvm::X86::EAX), IsImmediate(1)

                                   )))));
}

TEST_F(AsmParserTest, AssemblyFailure) {
  const auto result = ParseAsmCodeFromString(llvm_x86_->target_machine(),
                                             "this is not valid assembly",
                                             llvm::InlineAsm::AD_Intel);
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace gematria
