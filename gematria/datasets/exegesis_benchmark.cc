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
#include <optional>
#include <string>
#include <vector>

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
#include "llvm/tools/llvm-exegesis/lib/ResultAggregator.h"
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
  // TODO(boomanaiden154): Enable the usage of validation counters eventually
  const std::unique_ptr<BenchmarkRunner> Runner =
      ExitOnErr(State.getExegesisTarget().createBenchmarkRunner(
          Benchmark::Latency, State, BenchmarkPhaseSelectorE::Measure,
          BenchmarkRunner::ExecutionModeE::SubProcess, 30, {}, Benchmark::Min));

  if (pfm::pfmInitialize()) ExitWithError("Failed to initialize libpfm");

  for (const auto &AnnotatedBlock : *ParsedAnnotatedBlocks.getAsArray()) {
    std::optional<StringRef> HexValue =
        AnnotatedBlock.getAsObject()->getString("Hex");
    if (!HexValue) ExitWithError("Expected basic block to have hex value");

    std::optional<int64_t> LoopRegister =
        AnnotatedBlock.getAsObject()->getInteger("LoopRegister");
    if (!LoopRegister.has_value()) ExitWithError("Malfroemd basic block.");

    std::unique_ptr<const SnippetRepetitor> SnipRepetitor =
        SnippetRepetitor::Create(Benchmark::RepetitionModeE::MiddleHalfLoop,
                                 State, *LoopRegister);

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

    const json::Array *RegisterDefinitions =
        AnnotatedBlock.getAsObject()->getArray("RegisterDefinitions");

    for (const auto &RegisterDefinitionValue : *RegisterDefinitions) {
      const json::Object *RegisterDefinitionObject =
          RegisterDefinitionValue.getAsObject();

      RegisterValue RegVal;
      std::optional<int64_t> RegisterIndex =
          RegisterDefinitionObject->getInteger("Register");
      std::optional<int64_t> RegisterValue =
          RegisterDefinitionObject->getInteger("Value");
      if (!RegisterIndex.has_value() || !RegisterValue.has_value())
        ExitWithError("Malformed register definition");

      RegVal.Register = *RegisterIndex;
      RegVal.Value = APInt(64, *RegisterValue);
      BenchCode.Key.RegisterInitialValues.push_back(RegVal);
    }

    const json::Array *MemoryDefinitions =
        AnnotatedBlock.getAsObject()->getArray("MemoryDefinitions");

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
      // Might need to figure out something better for this as the value should
      // be an arbitrarily big integer.
      std::optional<int64_t> MemoryDefinitionHexValue =
          MemoryDefinitionObject->getInteger("Value");

      if (!MemoryDefinitionName.has_value() ||
          !MemoryDefinitionSize.has_value() ||
          !MemoryDefinitionHexValue.has_value())
        ExitWithError("Malformed memory definition");

      MemoryValue MemVal;
      MemVal.Value = APInt(32, *MemoryDefinitionHexValue);
      MemVal.Index = 0;  // Update this to support multiple definitions
      MemVal.SizeBytes = *MemoryDefinitionSize;

      BenchCode.Key.MemoryValues[MemoryDefinitionName->str()] = MemVal;
    }

    const json::Array *MemoryMappings =
        AnnotatedBlock.getAsObject()->getArray("MemoryMappings");

    if (!MemoryMappings) ExitWithError("Malformed memory mapping");

    for (const auto &MemoryMappingValue : *MemoryMappings) {
      const json::Object *MemoryMappingObject =
          MemoryMappingValue.getAsObject();

      if (!MemoryMappingObject) ExitWithError("Malformed memory mapping");

      std::optional<StringRef> MemoryMappingDefinitionName =
          MemoryMappingObject->getString("Value");
      std::optional<uintptr_t> MemoryMappingAddress =
          MemoryMappingObject->getInteger("Address");

      if (!MemoryMappingDefinitionName.has_value() ||
          !MemoryMappingAddress.has_value())
        ExitWithError("Malformed memory mapping");

      MemoryMapping MemMap;
      MemMap.Address = *MemoryMappingAddress;
      MemMap.MemoryValueName = MemoryMappingDefinitionName->str();
      BenchCode.Key.MemoryMappings.push_back(MemMap);
    }

    SmallVector<Benchmark, 2> AllResults;

    BenchmarkRunner::RunnableConfiguration RC1 = ExitOnErr(
        Runner->getRunnableConfiguration(BenchCode, 5000, 0, *SnipRepetitor));
    BenchmarkRunner::RunnableConfiguration RC2 = ExitOnErr(
        Runner->getRunnableConfiguration(BenchCode, 10000, 0, *SnipRepetitor));

    std::pair<Error, Benchmark> BenchmarkResult1OrErr =
        Runner->runConfiguration(std::move(RC1), {});

    if (std::get<0>(BenchmarkResult1OrErr)) {
      ExitOnErr(std::move(std::get<0>(BenchmarkResult1OrErr)));
    }

    AllResults.push_back(std::move(std::get<1>(BenchmarkResult1OrErr)));

    std::pair<Error, Benchmark> BenchmarkResult2OrErr =
        Runner->runConfiguration(std::move(RC2), {});

    if (std::get<0>(BenchmarkResult2OrErr))
      ExitOnErr(std::move(std::get<0>(BenchmarkResult2OrErr)));

    AllResults.push_back(std::move(std::get<1>(BenchmarkResult2OrErr)));

    std::unique_ptr<ResultAggregator> ResultAgg =
        ResultAggregator::CreateAggregator(
            Benchmark::RepetitionModeE::MiddleHalfLoop);

    Benchmark Result = std::move(AllResults[0]);

    ResultAgg->AggregateResults(Result,
                                ArrayRef<Benchmark>(AllResults).drop_front());

    unsigned Throughput100 = static_cast<unsigned>(
        round(Result.Measurements[0].PerSnippetValue * 100));

    dbgs() << *HexValue << "," << Throughput100 << "\n";
  }

  return 0;
}
