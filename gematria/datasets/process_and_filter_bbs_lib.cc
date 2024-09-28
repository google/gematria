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

#include <string>
#include <system_error>
#include <vector>

#include "gematria/llvm/disassembler.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/utils/string.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "llvm/include/llvm/ADT/Twine.h"
#include "llvm/include/llvm/MC/MCInstrDesc.h"
#include "llvm/include/llvm/Support/Error.h"
#include "llvm/lib/Target/X86/MCTargetDesc/X86MCTargetDesc.h"

using namespace llvm;

namespace gematria {

BBProcessorFilter::BBProcessorFilter()
    : LLVMSupport(LlvmArchitectureSupport::X86_64()),
      InstructionPrinter(LLVMSupport->CreateMCInstPrinter(0)) {}

Expected<std::string> BBProcessorFilter::removeRiskyInstructions(
    const StringRef BasicBlock, const StringRef Filename,
    bool FilterMemoryAccessingBlocks) {
  // TODO(boomanaiden154): Update this to use llvm::Expected once
  // gematria::ParseHex is refactored to return llvm::Expected.
  auto MachineCodeHex = gematria::ParseHexString(BasicBlock);
  if (!MachineCodeHex.has_value()) {
    return createFileError(
        Filename,
        make_error<StringError>(
            llvm::Twine("Could not parse: '").concat(BasicBlock).concat("'"),
            std::make_error_code(std::errc::invalid_argument)));
  }

  Expected<std::vector<gematria::DisassembledInstruction>>
      DisassembledInstructionsOrErr = gematria::DisassembleAllInstructions(
          LLVMSupport->mc_disassembler(), LLVMSupport->mc_instr_info(),
          LLVMSupport->mc_register_info(), LLVMSupport->mc_subtarget_info(),
          *InstructionPrinter, 0, *MachineCodeHex);

  if (!DisassembledInstructionsOrErr)
    return createFileError(Filename, DisassembledInstructionsOrErr.takeError());

  std::string OutputBlock;

  for (const gematria::DisassembledInstruction &Instruction :
       *DisassembledInstructionsOrErr) {
    MCInstrDesc InstDesc =
        LLVMSupport->mc_instr_info().get(Instruction.mc_inst.getOpcode());
    // Filter out syscalls as they do not match our benchmarking assumptions
    // and have potential security concerns. We filter out the three main
    // types of syscalls here:
    // 1. Software interrupts with int.
    // 2. Fast system calls with sysenter/sysexit.
    // 3. Fast system calls with syscall/sysret.
    // Call gates are also possible, but require far calls, which we filter out
    // below. We include the kernel side return instructions in case we pull
    // benchmarking code from privileged sources (like the Linux kernel).
    switch (Instruction.mc_inst.getOpcode()) {
      case X86::INT:
        continue;
      case X86::SYSENTER:
        continue;
      case X86::SYSEXIT:
        continue;
      case X86::SYSEXIT64:
        continue;
      case X86::SYSCALL:
        continue;
      case X86::SYSRET:
        continue;
      case X86::SYSRET64:
        continue;
    }

    if (InstDesc.isReturn() || InstDesc.isCall() || InstDesc.isBranch())
      continue;

    if (FilterMemoryAccessingBlocks &&
        (InstDesc.mayLoad() || InstDesc.mayStore()))
      continue;

    OutputBlock += toHex(Instruction.machine_code);
  }

  return OutputBlock;
}

}  // namespace gematria
