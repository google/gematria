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

#include "gematria/datasets/process_and_filter_bbs_lib.h"

#include "X86InstrInfo.h"
#include "absl/log/check.h"
#include "gematria/llvm/asm_parser.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/utils/string.h"
#include "gtest/gtest.h"
#include "llvm/MC/MCCodeEmitter.h"

using namespace llvm;

namespace gematria {
namespace {

class ProcessFilterBBsTest : public testing::Test {
 private:
  std::unique_ptr<LlvmArchitectureSupport> LLVMArchSupport;
  std::unique_ptr<MCInstPrinter> LLVMInstPrinter;
  BBProcessorFilter BBProcessor;

 protected:
  ProcessFilterBBsTest() : BBProcessor(){};

  void SetUp() override {
    LLVMArchSupport = LlvmArchitectureSupport::X86_64();
    LLVMInstPrinter =
        LLVMArchSupport->CreateMCInstPrinter(InlineAsm::AsmDialect::AD_ATT);
  }

  std::vector<DisassembledInstruction> removeRiskyInstructions(
      std::string_view TextualAssembly, bool FilterMemoryAccessingBlocks) {
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

    std::string BlockHex = FormatAsHexString(Code.str());
    Expected<std::string> ProcessedBlockHex =
        BBProcessor.removeRiskyInstructions(BlockHex, "test",
                                            FilterMemoryAccessingBlocks);
    CHECK(static_cast<bool>(ProcessedBlockHex));

    std::optional<std::vector<uint8_t>> BytesString =
        ParseHexString(*ProcessedBlockHex);
    CHECK(BytesString.has_value());

    Expected<std::vector<DisassembledInstruction>> InstructionsOrErr =
        DisassembleAllInstructions(LLVMArchSupport->mc_disassembler(),
                                   LLVMArchSupport->mc_instr_info(),
                                   LLVMArchSupport->mc_register_info(),
                                   LLVMArchSupport->mc_subtarget_info(),
                                   *LLVMInstPrinter, 0, *BytesString);
    CHECK(static_cast<bool>(InstructionsOrErr));
    return *InstructionsOrErr;
  }
};

// TODO(boomanaiden154): The formatting of the below removeRiskyInstructions
// function calls is weird. This appears to be a bug in clang-format that will
// hopefully be fixed before the next version bump.
// https://github.com/llvm/llvm-project/issues/100944

TEST_F(ProcessFilterBBsTest, RemoveSyscall) {
  auto ProcessedInstructions = removeRiskyInstructions(R"asm(
    movq %r11, %r12
    syscall
  )asm",
                                                       false);

  EXPECT_EQ(ProcessedInstructions.size(), 1);
  EXPECT_EQ(ProcessedInstructions[0].mc_inst.getOpcode(), X86::MOV64rr);
}

TEST_F(ProcessFilterBBsTest, RemoveReturn) {
  auto ProcessedInstructions = removeRiskyInstructions(R"asm(
    movq %r11, %r12
    retq
  )asm",
                                                       false);

  EXPECT_EQ(ProcessedInstructions.size(), 1);
  EXPECT_EQ(ProcessedInstructions[0].mc_inst.getOpcode(), X86::MOV64rr);
}

TEST_F(ProcessFilterBBsTest, RemoveCall) {
  auto ProcessedInstructions = removeRiskyInstructions(R"asm(
    movq %r11, %r12
    callq *%rax
  )asm",
                                                       false);

  EXPECT_EQ(ProcessedInstructions.size(), 1);
  EXPECT_EQ(ProcessedInstructions[0].mc_inst.getOpcode(), X86::MOV64rr);
}

TEST_F(ProcessFilterBBsTest, RemoveBranch) {
  auto ProcessedInstructions = removeRiskyInstructions(R"asm(
    movq %r11, %r12
    je 0x10
  )asm",
                                                       false);

  EXPECT_EQ(ProcessedInstructions.size(), 1);
  EXPECT_EQ(ProcessedInstructions[0].mc_inst.getOpcode(), X86::MOV64rr);
}

TEST_F(ProcessFilterBBsTest, RemoveMemoryAccessingInstructions) {
  auto ProcessedInstructions = removeRiskyInstructions(R"asm(
    movq %r11, %r12
    movq (%rax), %rax
  )asm",
                                                       true);

  EXPECT_EQ(ProcessedInstructions.size(), 1);
  EXPECT_EQ(ProcessedInstructions[0].mc_inst.getOpcode(), X86::MOV64rr);
}

TEST_F(ProcessFilterBBsTest, PreserveMemoryAccessingInstructions) {
  auto ProcessedInstructions = removeRiskyInstructions(R"asm(
    movq %r11, %r12
    movq (%rax), %rax
  )asm",
                                                       false);

  EXPECT_EQ(ProcessedInstructions.size(), 2);
  EXPECT_EQ(ProcessedInstructions[0].mc_inst.getOpcode(), X86::MOV64rr);
  EXPECT_EQ(ProcessedInstructions[1].mc_inst.getOpcode(), X86::MOV64rm);
}

}  // namespace
}  // namespace gematria
