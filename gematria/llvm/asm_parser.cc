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
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gematria/llvm/diagnostics.h"
#include "llvm/include/llvm/BinaryFormat/ELF.h"
#include "llvm/include/llvm/IR/InlineAsm.h"
#include "llvm/include/llvm/MC/MCContext.h"
#include "llvm/include/llvm/MC/MCDirectives.h"
#include "llvm/include/llvm/MC/MCInst.h"
#include "llvm/include/llvm/MC/MCObjectFileInfo.h"
#include "llvm/include/llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/include/llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/include/llvm/MC/MCSection.h"
#include "llvm/include/llvm/MC/MCSectionELF.h"
#include "llvm/include/llvm/MC/MCStreamer.h"
#include "llvm/include/llvm/MC/MCSubtargetInfo.h"
#include "llvm/include/llvm/MC/MCSymbol.h"
#include "llvm/include/llvm/MC/MCTargetOptions.h"
#include "llvm/include/llvm/MC/TargetRegistry.h"
#include "llvm/include/llvm/Support/Alignment.h"
#include "llvm/include/llvm/Support/MemoryBuffer.h"
#include "llvm/include/llvm/Support/SMLoc.h"
#include "llvm/include/llvm/Support/SourceMgr.h"
#include "llvm/include/llvm/Target/TargetMachine.h"

namespace gematria {
namespace {

// A streamer that puts MCInst's in a vector.
class MCInstStreamer : public llvm::MCStreamer {
 public:
  explicit MCInstStreamer(llvm::MCContext* context)
      : llvm::MCStreamer(*context),
        section_(context->getELFNamedSection("", "fake_section",
                                             llvm::ELF::SHT_PROGBITS, 0)) {}

  void emitInstruction(
      const llvm::MCInst& instruction,
      const llvm::MCSubtargetInfo& mc_subtarget_info) override {
    instructions_.push_back(instruction);
  }

  void initSections(bool, const llvm::MCSubtargetInfo&) override {
    switchSection(section_);
  }

  std::vector<llvm::MCInst> Get() && { return std::move(instructions_); }

 private:
  // We only care about instructions, we don't implement this part of the API.
  void emitCommonSymbol(llvm::MCSymbol* symbol, uint64_t size,
                        llvm::Align byte_alignment) override {}
  bool emitSymbolAttribute(llvm::MCSymbol* symbol,
                           llvm::MCSymbolAttr attribute) override {
    return false;
  }
  void emitValueToAlignment(llvm::Align alignment, int64_t value,
                            unsigned value_size,
                            unsigned max_bytes_to_emit) override {}
  void emitZerofill(llvm::MCSection* section, llvm::MCSymbol* symbol,
                    uint64_t size, llvm::Align byte_alignment,
                    llvm::SMLoc Loc) override {}

  std::vector<llvm::MCInst> instructions_;
  llvm::MCSectionELF* const section_;
};

}  // namespace

absl::StatusOr<std::vector<llvm::MCInst>> ParseAsmCodeFromBuffer(
    const llvm::TargetMachine& target_machine,
    std::unique_ptr<llvm::MemoryBuffer> buffer,
    const llvm::InlineAsm::AsmDialect dialect) {
  const llvm::Target& target = target_machine.getTarget();

  llvm::SourceMgr source_manager;
  source_manager.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

  llvm::MCTargetOptions options;
  llvm::MCContext mc_context(
      target_machine.getTargetTriple(), target_machine.getMCAsmInfo(),
      target_machine.getMCRegisterInfo(), target_machine.getMCSubtargetInfo(),
      &source_manager, &options);
  llvm::MCObjectFileInfo object_file_info;
  object_file_info.initMCObjectFileInfo(mc_context, /*PIC*/ true);
  mc_context.setObjectFileInfo(&object_file_info);

  MCInstStreamer streamer(&mc_context);
  const std::unique_ptr<llvm::MCAsmParser> asm_parser(llvm::createMCAsmParser(
      source_manager, mc_context, streamer, *target_machine.getMCAsmInfo()));
  asm_parser->setAssemblerDialect(dialect);

  const std::unique_ptr<llvm::MCTargetAsmParser> target_asm_parser(
      target.createMCAsmParser(*target_machine.getMCSubtargetInfo(),
                               *asm_parser, *target_machine.getMCInstrInfo(),
                               options));

  if (!target_asm_parser) {
    return absl::InternalError("cannot create target asm parser");
  }
  asm_parser->setTargetParser(*target_asm_parser);

  // Intercept errors to put them in the returned status in case of failure.
  ScopedStringDiagnosticHandler errors(mc_context);
  if (asm_parser->Run(false)) {
    return absl::InvalidArgumentError(
        absl::StrCat("cannot parse asm file:\n", errors.Get()));
  }
  return std::move(streamer).Get();
}

absl::StatusOr<std::vector<llvm::MCInst>> ParseAsmCodeFromString(
    const llvm::TargetMachine& target_machine, std::string_view assembly,
    llvm::InlineAsm::AsmDialect dialect) {
  std::unique_ptr<llvm::MemoryBuffer> buffer =
      llvm::MemoryBuffer::getMemBuffer(assembly);
  return ParseAsmCodeFromBuffer(target_machine, std::move(buffer), dialect);
}

}  // namespace gematria
