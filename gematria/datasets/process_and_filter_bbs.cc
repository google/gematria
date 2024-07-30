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
#include <limits>

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86RegisterInfo.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/utils/string.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"

using namespace llvm;

cl::OptionCategory ProcessFilterCat("process_and_filter_bbs options",
                                    "The options specifically for controlling "
                                    "the behavior of process_and_filter_bbs.");

static cl::opt<std::string> InputFile(
    "input-file",
    cl::desc("Path to the input CSV file containing hex basic blocks"),
    cl::init(""), cl::cat(ProcessFilterCat));

static cl::opt<std::string> OutputFile(
    "output-file",
    cl::desc(
        "Path to the output CSV file with processed/filtered basic blocks"),
    cl::init(""), cl::cat(ProcessFilterCat));

static cl::opt<bool> FilterMemoryAccessingBlocks(
    "filter-memory-accessing-blocks",
    cl::desc("Whether or not to filter out blocks that access memory"),
    cl::init(false), cl::cat(ProcessFilterCat));

static cl::opt<unsigned> ReportProgressEvery(
    "report-progress-every",
    cl::desc("The interval at which to report progress in blocks"),
    cl::init(std::numeric_limits<unsigned>::max()), cl::cat(ProcessFilterCat));

static cl::opt<bool> FilterVariableLatencyInstructions(
    "filter-variable-latency-instructions",
    cl::desc("Whether or not to filter variable latency instructions"),
    cl::init(false), cl::cat(ProcessFilterCat));

Expected<std::string> processBasicBlock(
    const std::string &BasicBlock,
    const gematria::LlvmArchitectureSupport &LLVMSupport,
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
          LLVMSupport.mc_disassembler(), LLVMSupport.mc_instr_info(),
          LLVMSupport.mc_register_info(), LLVMSupport.mc_subtarget_info(),
          MachineInstructionPrinter, 0, *MachineCodeHex);

  if (!DisassembledInstructionsOrErr)
    return createFileError(FileName, DisassembledInstructionsOrErr.takeError());

  std::string OutputBlock;

  for (const gematria::DisassembledInstruction &Instruction :
       *DisassembledInstructionsOrErr) {
    MCInstrDesc InstDesc =
        LLVMSupport.mc_instr_info().get(Instruction.mc_inst.getOpcode());
    // TODO(boomanaiden154): This filtering is a bit simplistic currently. We
    // should probably be using MCInsrtDesc::hasUnmodeledSideEffects, but this
    // needs to be evaluated at scale.
    if (Instruction.mc_inst.getOpcode() == X86::SYSCALL) continue;
    if (InstDesc.isReturn() || InstDesc.isCall() || InstDesc.isBranch())
      continue;
    if (FilterMemoryAccessingBlocks &&
        (InstDesc.mayLoad() || InstDesc.mayStore()))
      continue;

    if (FilterVariableLatencyInstructions) {
      // Kind of a hacky solution, but can't figure out a better way to get a
      // list of all the variable latency instructions, and div seems like the
      // main one currently.
      if (Instruction.assembly.find("DIV") != std::string::npos)
        continue;
    }

    OutputBlock += toHex(Instruction.machine_code);
  }

  return OutputBlock;
}

int main(int Argc, char **Argv) {
  cl::ParseCommandLineOptions(Argc, Argv, "process_and_filter_bbs");

  ExitOnError ExitOnErr("process_and_filter_bbs error: ");

  const std::unique_ptr<gematria::LlvmArchitectureSupport> LLVMSupport =
      gematria::LlvmArchitectureSupport::X86_64();

  std::unique_ptr<MCInstPrinter> MachineInstructionPrinter =
      LLVMSupport->CreateMCInstPrinter(0);

  unsigned LineCount = 0;

  std::ifstream InputFileStream(InputFile);
  std::ofstream OutputFileStream(OutputFile);
  for (std::string Line; std::getline(InputFileStream, Line);) {
    Expected<std::string> ProcessedBlockOrErr = processBasicBlock(
        Line, *LLVMSupport, *MachineInstructionPrinter, InputFile);
    if (!ProcessedBlockOrErr) ExitOnErr(ProcessedBlockOrErr.takeError());

    OutputFileStream << *ProcessedBlockOrErr << "\n";

    if (LineCount != 0 && LineCount % ReportProgressEvery == 0)
      dbgs() << "Finished block " << LineCount << "\n";

    ++LineCount;
  }
  return 0;
}
