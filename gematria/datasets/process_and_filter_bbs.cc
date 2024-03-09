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

static cl::opt<bool> LastInstructionIsTerminator(
    "last-instruction-is-terminator",
    cl::desc("Whether or not the last instruction changes the control flow"),
    cl::init(true));

static ExitOnError ExitOnErr("process_and_filter_bbs error: ");

static void ExitOnFileError(const Twine &FileName, Error Err) {
  if (Err) {
    ExitOnErr(createFileError(FileName, std::move(Err)));
  }
}

int main(int Argc, char **Argv) {
  cl::ParseCommandLineOptions(Argc, Argv, "process_and_filter_bbs");

  const std::unique_ptr<gematria::LlvmArchitectureSupport> llvm_support =
      gematria::LlvmArchitectureSupport::X86_64();

  std::unique_ptr<MCInstPrinter> MachineInstructionPrinter =
      llvm_support->CreateMCInstPrinter(0);

  std::ifstream InputFileStream(InputFile);
  std::ofstream OutputFileStream(OutputFile);
  for (std::string Line; std::getline(InputFileStream, Line);) {
    auto MachineCodeHex = gematria::ParseHexString(Line);
    if (!MachineCodeHex.has_value()) {
      dbgs() << "could not parse: " << Line << "\n";
      return 3;
    }

    std::vector<gematria::DisassembledInstruction> DisassembledInstructions =
        ExitOnErr(gematria::DisassembleAllInstructions(
            llvm_support->mc_disassembler(), llvm_support->mc_instr_info(),
            llvm_support->mc_register_info(), llvm_support->mc_subtarget_info(),
            *MachineInstructionPrinter, 0, *MachineCodeHex));

    if (LastInstructionIsTerminator) DisassembledInstructions.pop_back();

    for (const gematria::DisassembledInstruction &Instruction :
         DisassembledInstructions) {
      OutputFileStream << toHex(Instruction.machine_code);
    }
    OutputFileStream << "\n";
  }
  return 0;
}
