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

#include "gematria/datasets/bhive_to_exegesis.h"

#include "absl/log/check.h"
#include "gematria/llvm/asm_parser.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/utils/string.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/tools/llvm-exegesis/lib/TargetSelect.h"

using namespace llvm;
using namespace llvm::exegesis;

using testing::SizeIs;

namespace gematria {
namespace {

class BHiveToExegesisTest : public testing::Test {
 private:
  std::unique_ptr<LlvmArchitectureSupport> LLVMArchSupport;
  std::unique_ptr<BHiveToExegesis> BHiveAnnotator;

 protected:
  BHiveToExegesisTest()
      : LLVMArchSupport(LlvmArchitectureSupport::X86_64()),
        BHiveAnnotator(cantFail(BHiveToExegesis::create(*LLVMArchSupport))) {}

  static void SetUpTestSuite() { InitializeX86ExegesisTarget(); }

  std::string AssembleToHex(std::string_view TextualAssembly) {
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

    return FormatAsHexString(Code.str());
  }

  absl::StatusOr<AnnotatedBlock> AnnotateBlock(
      std::string_view TextualAssembly, const unsigned MaxAnnotationAttempts,
      BHiveToExegesis::AnnotatorType AnnotatorToUse) {
    std::string BBHex = AssembleToHex(TextualAssembly);
    return BHiveAnnotator->annotateBasicBlock(BBHex, AnnotatorToUse,
                                              MaxAnnotationAttempts);
  }
};

TEST_F(BHiveToExegesisTest, SimpleBlock) {
  absl::StatusOr<AnnotatedBlock> BlockAnnotations =
      AnnotateBlock(R"asm(
    movq %r11, %r12
  )asm",
                    50, BHiveToExegesis::AnnotatorType::kFast);

  EXPECT_EQ(BlockAnnotations->BasicBlockProto.machine_instructions_size(), 1);

  EXPECT_THAT(BlockAnnotations->AccessedAddrs.accessed_blocks, SizeIs(0));
  EXPECT_THAT(BlockAnnotations->AccessedAddrs.initial_regs, SizeIs(1));
  EXPECT_TRUE(BlockAnnotations->AccessedAddrs.loop_register.has_value());
}

}  // namespace
}  // namespace gematria
