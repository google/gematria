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

#include <iostream>

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86RegisterInfo.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/utils/string.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkRunner.h"
#include "llvm/tools/llvm-exegesis/lib/LlvmState.h"
#include "llvm/tools/llvm-exegesis/lib/Target.h"
#include "llvm/tools/llvm-exegesis/lib/TargetSelect.h"

using namespace llvm;
using namespace llvm::exegesis;

static cl::opt<std::string> AnnotatedBlocksJson(
    "annotated-blocks-json",
    cl::desc("Filename of the JSON file containing annotated basic blocks"),
    cl::init(""));

static ExitOnError ExitOnErr("exegesis-benchmark error: ");

static void ExitWithError(StringRef ErrorMessage) {
  ExitOnErr(make_error<StringError>(ErrorMessage, inconvertibleErrorCode()));
}

static void ExitOnFileError(const Twine &FileName, Error Err) {
  if (Err) {
    ExitOnErr(createFileError(FileName, std::move(Err)));
  }
}

template <typename T>
T ExitOnFileError(const Twine &FileName, Expected<T> &&E) {
  ExitOnFileError(FileName, E.takeError());
  return std::move(*E);
}

int main(int Argc, char *Argv[]) {
  cl::ParseCommandLineOptions(
      Argc, Argv, "Tool for benchmarking sets of annotated basic blocks");
  if (AnnotatedBlocksJson.empty()) {
    ExitWithError("--annotated_blocks_json is required");
    return 1;
  }

  auto JsonMemoryBuffer = ExitOnFileError(
      AnnotatedBlocksJson,
      errorOrToExpected(MemoryBuffer::getFile(AnnotatedBlocksJson, true)));

  auto ParsedAnnotatedBlocks =
      ExitOnErr(json::parse(JsonMemoryBuffer->getBuffer()));

  // LLVM Setup
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86Target();
  LLVMInitializeX86AsmPrinter();
  LLVMInitializeX86AsmParser();
  LLVMInitializeX86Disassembler();

  // Exegesis Setup
  InitializeX86ExegesisTarget();

  const LLVMState State = ExitOnErr(LLVMState::Create("", "native"));

  // More LLVM Setup
  std::unique_ptr<MCContext> MachineContext = std::make_unique<MCContext>(
      State.getTargetMachine().getTargetTriple(),
      State.getTargetMachine().getMCAsmInfo(), &State.getRegInfo(),
      &State.getSubtargetInfo());

  std::unique_ptr<MCDisassembler> MachineDisassembler(
      State.getTargetMachine().getTarget().createMCDisassembler(
          State.getSubtargetInfo(), *MachineContext));

  std::unique_ptr<MCInstPrinter> MachinePrinter(
      State.getTargetMachine().getTarget().createMCInstPrinter(
          State.getTargetMachine().getTargetTriple(), 0,
          *State.getTargetMachine().getMCAsmInfo(), State.getInstrInfo(),
          State.getRegInfo()));

  // More exegesis setup
  // TODO(boomanaiden154): Switch to middle-half-loop eventually
  const std::unique_ptr<BenchmarkRunner> Runner =
      ExitOnErr(State.getExegesisTarget().createBenchmarkRunner(
          Benchmark::Latency, State, BenchmarkPhaseSelectorE::Measure,
          BenchmarkRunner::ExecutionModeE::SubProcess, 30, {}, Benchmark::Min));

  std::unique_ptr<const SnippetRepetitor> SnipRepetitor =
      SnippetRepetitor::Create(Benchmark::RepetitionModeE::Loop, State);

  if (pfm::pfmInitialize()) ExitWithError("Failed to initialize libpfm");

  for (const auto &TestValue : *ParsedAnnotatedBlocks.getAsArray()) {
    std::optional<StringRef> HexValue =
        TestValue.getAsObject()->getString("Hex");
    if (!HexValue) ExitWithError("Expected basic block to have hex value");

    std::optional<std::vector<uint8_t>> BytesOr =
        gematria::ParseHexString(HexValue->str());

    if (!BytesOr.has_value()) ExitWithError("Failed to parse hex value");

    std::vector<gematria::DisassembledInstruction> DisInstructions =
        ExitOnErr(gematria::DisassembleAllInstructions(
            *MachineDisassembler, State.getInstrInfo(), State.getRegInfo(),
            State.getSubtargetInfo(), *MachinePrinter, 0, *BytesOr));

    std::vector<MCInst> Instructions;
    Instructions.reserve(DisInstructions.size());

    for (const auto &DisInstruction : DisInstructions)
      Instructions.push_back(DisInstruction.mc_inst);

    BenchmarkCode BenchCode;
    BenchCode.Key.Instructions = std::move(Instructions);

    const llvm::MCRegisterInfo &MRI = State.getRegInfo();

    for (unsigned I = 0;
         I < MRI.getRegClass(X86::GR64_NOREX2RegClassID).getNumRegs(); ++I) {
      RegisterValue RegVal;
      RegVal.Register =
          MRI.getRegClass(X86::GR64_NOREX2RegClassID).getRegister(I);
      RegVal.Value = APInt(64, 0x12345600);
      BenchCode.Key.RegisterInitialValues.push_back(RegVal);
    }

    const json::Array *MemoryDefinitions =
        TestValue.getAsObject()->getArray("MemoryDefinitions");

    if (!MemoryDefinitions)
      ExitWithError("Expected field MemoryDefinitions does not exist.");

    for (const auto &MemoryDefinitionValue : *MemoryDefinitions) {
      const json::Object *MemoryDefinitionObject =
          MemoryDefinitionValue.getAsObject();

      if (!MemoryDefinitionObject) {
        ExitWithError("Malformed memory definition");
      }

      std::optional<StringRef> MemoryDefinitionName =
          MemoryDefinitionObject->getString("Name");
      std::optional<int64_t> MemoryDefinitionSize =
          MemoryDefinitionObject->getInteger("Size");
      std::optional<uintptr_t> MemoryDefinitionHexValue =
          MemoryDefinitionObject->getInteger("Value");

      if (!MemoryDefinitionName.has_value() ||
          !MemoryDefinitionSize.has_value() ||
          !MemoryDefinitionHexValue.has_value())
        ExitWithError("Malformed memory definition");

      MemoryValue MemVal;
      MemVal.Value = APInt(64, *MemoryDefinitionHexValue);
      MemVal.Index = 0;  // Update this to support multiple definitions
      MemVal.SizeBytes = *MemoryDefinitionSize;

      BenchCode.Key.MemoryValues[MemoryDefinitionName->str()] = MemVal;
    }

    BenchmarkRunner::RunnableConfiguration RC = ExitOnErr(
        Runner->getRunnableConfiguration(BenchCode, 10000, 0, *SnipRepetitor));

    std::pair<Error, Benchmark> BenchmarkResultOrErr =
        Runner->runConfiguration(std::move(RC), {});

    if (std::get<0>(BenchmarkResultOrErr)) {
      ExitOnErr(std::move(std::get<0>(BenchmarkResultOrErr)));
    }

    Benchmark Bench = std::move(std::get<1>(BenchmarkResultOrErr));
    dbgs() << Bench.Measurements[0].PerSnippetValue << "\n";

    dbgs() << *HexValue << "\n";

    break;
  }

  return 0;
}
