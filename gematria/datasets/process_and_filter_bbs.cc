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

#include <fstream>

#include "gematria/llvm/disassembler.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/utils/string.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"

using namespace llvm;

static cl::opt<std::string> InputFile(
    "input-file",
    cl::desc("Path to the input CSV file containing hex basic blocks"),
    cl::init(""));

static cl::opt<std::string> OutputFile(
    "output-file",
    cl::desc(
        "Path to the output CSV file with processed/filtered basic blocks"),
    cl::init(""));

Expected<std::string> ProcessBasicBlock(
    const std::string &BasicBlock,
    const gematria::LlvmArchitectureSupport &llvm_support,
    MCInstPrinter &MachineInstructionPrinter, const StringRef FileName) {
  // TODO(boomanaiden154): Update this to use llvm::Expected once
  // gematria::ParseHex is refactored to return llvm::Expected.
  auto MachineCodeHex = gematria::ParseHexString(BasicBlock);
  if (!MachineCodeHex.has_value()) {
    return createFileError(
        FileName,
        make_error<StringError>(
            llvm::Twine("Could not parse: '").concat(BasicBlock).concat("'"),
            std::make_error_code(std::errc::invalid_argument)));
  }

  Expected<std::vector<gematria::DisassembledInstruction>>
      DisassembledInstructionsOrErr = gematria::DisassembleAllInstructions(
          llvm_support.mc_disassembler(), llvm_support.mc_instr_info(),
          llvm_support.mc_register_info(), llvm_support.mc_subtarget_info(),
          MachineInstructionPrinter, 0, *MachineCodeHex);

  if (!DisassembledInstructionsOrErr)
    return createFileError(FileName, DisassembledInstructionsOrErr.takeError());

  std::string OutputBlock;

  for (const gematria::DisassembledInstruction &Instruction :
       *DisassembledInstructionsOrErr) {
    MCInstrDesc InstDesc =
        llvm_support.mc_instr_info().get(Instruction.mc_inst.getOpcode());
    if (InstDesc.isReturn() || InstDesc.isCall() || InstDesc.isBranch())
      continue;
    OutputBlock += toHex(Instruction.machine_code);
  }

  return OutputBlock;
}

int main(int Argc, char **Argv) {
  cl::ParseCommandLineOptions(Argc, Argv, "process_and_filter_bbs");

  ExitOnError ExitOnErr("process_and_filter_bbs error: ");

  const std::unique_ptr<gematria::LlvmArchitectureSupport> llvm_support =
      gematria::LlvmArchitectureSupport::X86_64();

  std::unique_ptr<MCInstPrinter> MachineInstructionPrinter =
      llvm_support->CreateMCInstPrinter(0);

  std::ifstream InputFileStream(InputFile);
  std::ofstream OutputFileStream(OutputFile);
  for (std::string Line; std::getline(InputFileStream, Line);) {
    Expected<std::string> ProcessedBlockOrErr = ProcessBasicBlock(
        Line, *llvm_support, *MachineInstructionPrinter, InputFile);
    if (!ProcessedBlockOrErr) ExitOnErr(ProcessedBlockOrErr.takeError());

    OutputFileStream << *ProcessedBlockOrErr << "\n";
  }
  return 0;
}
