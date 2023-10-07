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

#ifndef THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_STATE_H_
#define THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_STATE_H_

#include <memory>
#include <string_view>

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/TargetMachine.h"

namespace gematria {

// Provides a single handle to all LLVM objects representing a given
// architecture that can be passed around easily and shared with Python code.
class LlvmArchitectureSupport {
 public:
  // Creates the architecture support from an LLVM triple. Returns an error when
  // the architecture can't be created.
  static llvm::Expected<std::unique_ptr<LlvmArchitectureSupport>> FromTriple(
      std::string_view llvm_triple, std::string_view cpu,
      std::string_view cpu_features);

  // A convenience function that creates the architecture support for x86-64.
  // Calls the necessary LLVMInitializeX86*() functions on the first invocation.
  static std::unique_ptr<LlvmArchitectureSupport> X86_64();

  // Creates a new llvm::MCInstPriner. The value of `syntax_variant` is
  // architecture dependent, and corresponds to the same argument of
  // createMCInstPrinter.
  // In most cases, 0 means the "AT&T syntax". On x86-64, 1 is "Intel syntax".
  std::unique_ptr<llvm::MCInstPrinter> CreateMCInstPrinter(
      int syntax_variant) const {
    return std::unique_ptr<llvm::MCInstPrinter>(target_->createMCInstPrinter(
        target_machine_->getTargetTriple(), syntax_variant, mc_asm_info(),
        mc_instr_info(), mc_register_info()));
  }

  const llvm::Target& target() const { return *target_; }

  const llvm::TargetMachine& target_machine() const { return *target_machine_; }

  const llvm::MCAsmInfo& mc_asm_info() const {
    return *target_machine_->getMCAsmInfo();
  }

  const llvm::MCInstrInfo& mc_instr_info() const {
    return *target_machine_->getMCInstrInfo();
  }

  const llvm::MCRegisterInfo& mc_register_info() const {
    return *target_machine_->getMCRegisterInfo();
  }

  const llvm::MCDisassembler& mc_disassembler() const {
    return *mc_disassembler_;
  }

  const llvm::MCSubtargetInfo& mc_subtarget_info() const {
    return *target_machine_->getMCSubtargetInfo();
  }

 private:
  LlvmArchitectureSupport(std::string_view llvm_triple, std::string_view cpu,
                          std::string_view cpu_features,
                          const llvm::Target* llvm_target);

  const llvm::Target* target_;
  std::unique_ptr<llvm::TargetMachine> target_machine_;
  std::unique_ptr<llvm::MCContext> mc_context_;
  std::unique_ptr<llvm::MCDisassembler> mc_disassembler_;
};

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_STATE_H_
