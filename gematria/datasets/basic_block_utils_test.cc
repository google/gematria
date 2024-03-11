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
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/llvm/asm_parser.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/testing/matchers.h"

#include "gtest/gtest.h"

namespace gematria {
namespace {
class BasicBlockUtilsTest : public ::testing::Test {
private:
  static std::unique_ptr<LlvmArchitectureSupport> LlvmArchSupport;

protected:
  static void SetUpTestSuite() {
    LlvmArchSupport = LlvmArchitectureSupport::X86_64();
  }

  std::vector<DisassembledInstruction> getInstructions(std::string_view textual_assembly) {
    auto MCInstsOrError = gematria::ParseAsmCodeFromString(LlvmArchSupport->target_machine(),
        textual_assembly, llvm::InlineAsm::AsmDialect::AD_ATT);
    CHECK_OK(MCInstsOrError);

    std::vector<DisassembledInstruction> Instructions;
    Instructions.reserve(MCInstsOrError->size());

    for (MCInst Instruction : *MCInstsOrError)
      Instructions.push_back({0, "", "", Instruction});

    return Instructions;
  }

  std::vector<unsigned> getUsedRegisters(std::string_view textual_assembly) {
    return BasicBlockUtils::getUsedRegisters(getInstructions(textual_assembly),
        LlvmArchSupport->mc_register_info(), LlvmArchSupport->mc_instr_info());
  }
};

TEST_F(BasicBlockUtilsTest, GetUsedRegisters) {
  //std::vector<unsigned> UsedRegisters = BasicBlockUtils::GetUsedRegisters({});
  EXPECT_EQ({}, 0);
}

} //namespace
} // namespace gematria
