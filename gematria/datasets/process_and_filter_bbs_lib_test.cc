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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "gematria/llvm/asm_parser.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/utils/string.h"
#include "gtest/gtest.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Support/Error.h"
#include "llvm/lib/Target/X86/MCTargetDesc/X86MCTargetDesc.h"

using namespace llvm;

namespace gematria {
namespace {

class ProcessFilterBBsTest : public testing::Test {
 private:
  std::unique_ptr<LlvmArchitectureSupport> LLVMArchSupport;
  std::unique_ptr<MCInstPrinter> LLVMInstPrinter;
  BBProcessorFilter BBProcessor;

 protected:
  ProcessFilterBBsTest()
      : LLVMArchSupport(LlvmArchitectureSupport::X86_64()),
        LLVMInstPrinter(LLVMArchSupport->CreateMCInstPrinter(
            InlineAsm::AsmDialect::AD_ATT)),
        BBProcessor() {};

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

// TODO(boomanaiden154): THe assembly strings were moved out to
// constexpr local variables to get around formatting issues. When those are
// fixed (https://github.com/llvm/llvm-project/issues/100944), we should move
// the assembly back to a direct parameter of the functions.

TEST_F(ProcessFilterBBsTest, RemoveSoftwareInterrupt) {
  static constexpr std::string_view Assembly = R"asm(
    movq %r11, %r12
    int $0x80
  )asm";

  auto ProcessedInstructions = removeRiskyInstructions(Assembly, false);

  EXPECT_EQ(ProcessedInstructions.size(), 1);
  EXPECT_EQ(ProcessedInstructions[0].mc_inst.getOpcode(), X86::MOV64rr);
}

TEST_F(ProcessFilterBBsTest, RemoveSyscall) {
  static constexpr std::string_view Assembly = R"asm(
    movq %r11, %r12
    syscall
    sysret
    sysretq
  )asm";

  auto ProcessedInstructions = removeRiskyInstructions(Assembly, false);

  EXPECT_EQ(ProcessedInstructions.size(), 1);
  EXPECT_EQ(ProcessedInstructions[0].mc_inst.getOpcode(), X86::MOV64rr);
}

TEST_F(ProcessFilterBBsTest, RemoveSysenter) {
  static constexpr std::string_view Assembly = R"asm(
    movq %r11, %r12
    sysenter
    sysexit
    sysexitq
  )asm";

  auto ProcessedInstructions = removeRiskyInstructions(Assembly, false);

  EXPECT_EQ(ProcessedInstructions.size(), 1);
  EXPECT_EQ(ProcessedInstructions[0].mc_inst.getOpcode(), X86::MOV64rr);
}

TEST_F(ProcessFilterBBsTest, RemoveReturn) {
  static constexpr std::string_view Assembly = R"asm(
    movq %r11, %r12
    retq
  )asm";

  auto ProcessedInstructions = removeRiskyInstructions(Assembly, false);

  EXPECT_EQ(ProcessedInstructions.size(), 1);
  EXPECT_EQ(ProcessedInstructions[0].mc_inst.getOpcode(), X86::MOV64rr);
}

TEST_F(ProcessFilterBBsTest, RemoveCall) {
  static constexpr std::string_view Assembly = R"asm(
    movq %r11, %r12
    callq *%rax
  )asm";

  auto ProcessedInstructions = removeRiskyInstructions(Assembly, false);

  EXPECT_EQ(ProcessedInstructions.size(), 1);
  EXPECT_EQ(ProcessedInstructions[0].mc_inst.getOpcode(), X86::MOV64rr);
}

TEST_F(ProcessFilterBBsTest, RemoveBranch) {
  static constexpr std::string_view Assembly = R"asm(
    movq %r11, %r12
    je 0x10
  )asm";

  auto ProcessedInstructions = removeRiskyInstructions(Assembly, false);

  EXPECT_EQ(ProcessedInstructions.size(), 1);
  EXPECT_EQ(ProcessedInstructions[0].mc_inst.getOpcode(), X86::MOV64rr);
}

TEST_F(ProcessFilterBBsTest, RemoveMemoryAccessingInstructions) {
  static constexpr std::string_view Assembly = R"asm(
    movq %r11, %r12
    movq (%rax), %rax
  )asm";

  auto ProcessedInstructions = removeRiskyInstructions(Assembly, true);

  EXPECT_EQ(ProcessedInstructions.size(), 1);
  EXPECT_EQ(ProcessedInstructions[0].mc_inst.getOpcode(), X86::MOV64rr);
}

TEST_F(ProcessFilterBBsTest, PreserveMemoryAccessingInstructions) {
  static constexpr std::string_view Assembly = R"asm(
    movq %r11, %r12
    movq (%rax), %rax
  )asm";

  auto ProcessedInstructions = removeRiskyInstructions(Assembly, false);

  EXPECT_EQ(ProcessedInstructions.size(), 2);
  EXPECT_EQ(ProcessedInstructions[0].mc_inst.getOpcode(), X86::MOV64rr);
  EXPECT_EQ(ProcessedInstructions[1].mc_inst.getOpcode(), X86::MOV64rm);
}

}  // namespace
}  // namespace gematria
