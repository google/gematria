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
#include "llvm/Support/Errc.h"
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

Expected<BenchmarkCode> parseJSONBlock(
    const json::Object &BasicBlockJSON, MCInstPrinter &MachinePrinter,
    const MCDisassembler &MachineDisassembler, const LLVMState &State) {
  BenchmarkCode BenchCode;

  std::optional<StringRef> HexValue = BasicBlockJSON.getString("Hex");
  if (!HexValue)
    return llvm::make_error<StringError>(
        errc::invalid_argument, "Expected basic block to have hex value");

  std::optional<int64_t> LoopRegister =
      BasicBlockJSON.getInteger("LoopRegister");
  if (!LoopRegister.has_value())
    return llvm::make_error<StringError>(
        errc::invalid_argument, "Malformed basic block: no loop register");

  BenchCode.Key.LoopRegister = *LoopRegister;

  // TODO(ondrasej): Update this after converting gematria::ParseHexString to
  // return llvm::Expected rather than an optional.
  std::optional<std::vector<uint8_t>> BytesOr =
      gematria::ParseHexString(HexValue->str());

  if (!BytesOr.has_value())
    return llvm::make_error<StringError>(
        errc::invalid_argument, "Malformed basic block: invalid hex value");

  Expected<std::vector<gematria::DisassembledInstruction>>
      DisInstructionsOrErr = gematria::DisassembleAllInstructions(
          MachineDisassembler, State.getInstrInfo(), State.getRegInfo(),
          State.getSubtargetInfo(), MachinePrinter, 0, *BytesOr);

  if (!DisInstructionsOrErr) return DisInstructionsOrErr.takeError();

  std::vector<MCInst> Instructions;
  Instructions.reserve(DisInstructionsOrErr->size());

  for (const auto &DisInstruction : *DisInstructionsOrErr)
    Instructions.push_back(DisInstruction.mc_inst);

  BenchCode.Key.Instructions = std::move(Instructions);

  const json::Array *RegisterDefinitions =
      BasicBlockJSON.getArray("RegisterDefinitions");

  if (!RegisterDefinitions)
    return llvm::make_error<StringError>(
        errc::invalid_argument,
        "Malformed basic block: no register definitions array");

  for (const auto &RegisterDefinitionValue : *RegisterDefinitions) {
    const json::Object *RegisterDefinitionObject =
        RegisterDefinitionValue.getAsObject();

    RegisterValue RegVal;
    std::optional<int64_t> RegisterIndex =
        RegisterDefinitionObject->getInteger("Register");
    std::optional<int64_t> RegisterValue =
        RegisterDefinitionObject->getInteger("Value");
    if (!RegisterIndex.has_value() || !RegisterValue.has_value())
      return llvm::make_error<StringError>(
          errc::invalid_argument,
          "Malformed register definition: invalid register number or value");

    RegVal.Register = *RegisterIndex;
    RegVal.Value = APInt(64, *RegisterValue);
    BenchCode.Key.RegisterInitialValues.push_back(RegVal);
  }

  const json::Array *MemoryDefinitions =
      BasicBlockJSON.getArray("MemoryDefinitions");

  if (!MemoryDefinitions)
    return llvm::make_error<StringError>(
        errc::invalid_argument,
        "Malformed basic block: no memory definitions array");

  size_t MemoryDefinitionIndex = 0;

  for (const auto &MemoryDefinitionValue : *MemoryDefinitions) {
    const json::Object *MemoryDefinitionObject =
        MemoryDefinitionValue.getAsObject();

    if (!MemoryDefinitionObject)
      return llvm::make_error<StringError>(
          errc::invalid_argument,
          "Malformed memory definition: memory "
          "definition is not a JSON object");

    std::optional<StringRef> MemoryDefinitionName =
        MemoryDefinitionObject->getString("Name");
    std::optional<int64_t> MemoryDefinitionSize =
        MemoryDefinitionObject->getInteger("Size");

    std::optional<int64_t> MemoryDefinitionHexValue =
        MemoryDefinitionObject->getInteger("Value");

    if (!MemoryDefinitionName.has_value() ||
        !MemoryDefinitionSize.has_value() ||
        !MemoryDefinitionHexValue.has_value())
      return llvm::make_error<StringError>(
          errc::invalid_argument,
          "Malformed memory definition: no size, name, or value");

    MemoryValue MemVal;
    MemVal.Value = APInt(32, *MemoryDefinitionHexValue);
    MemVal.Index = MemoryDefinitionIndex;
    MemVal.SizeBytes = *MemoryDefinitionSize;

    BenchCode.Key.MemoryValues[MemoryDefinitionName->str()] = MemVal;

    ++MemoryDefinitionIndex;
  }

  const json::Array *MemoryMappings = BasicBlockJSON.getArray("MemoryMappings");

  if (!MemoryMappings)
    return llvm::make_error<StringError>(
        errc::invalid_argument,
        "Malformed basic block: no memory mappings array");

  for (const auto &MemoryMappingValue : *MemoryMappings) {
    const json::Object *MemoryMappingObject = MemoryMappingValue.getAsObject();

    if (!MemoryMappingObject)
      return llvm::make_error<StringError>(
          errc::invalid_argument,
          "Malformed memory mapping: memory mapping is not a JSON object");

    std::optional<StringRef> MemoryMappingDefinitionName =
        MemoryMappingObject->getString("Value");
    std::optional<uintptr_t> MemoryMappingAddress =
        MemoryMappingObject->getInteger("Address");

    if (!MemoryMappingDefinitionName.has_value() ||
        !MemoryMappingAddress.has_value())
      return llvm::make_error<StringError>(
          errc::invalid_argument,
          "Malformed memory mapping: no name or address");

    MemoryMapping MemMap;
    MemMap.Address = *MemoryMappingAddress;
    MemMap.MemoryValueName = MemoryMappingDefinitionName->str();
    BenchCode.Key.MemoryMappings.push_back(MemMap);
  }

  return BenchCode;
}

int main(int Argc, char *Argv[]) {
  cl::ParseCommandLineOptions(
      Argc, Argv, "Tool for benchmarking sets of annotated basic blocks");

  ExitOnError ExitOnErr("exegesis-benchmark error: ");
  if (AnnotatedBlocksJson.empty())
    ExitOnErr(llvm::make_error<StringError>(
        errc::invalid_argument, "--annotated_blocks_json is required"));

  auto JsonMemoryBuffer = ExitOnErr(
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

  // TODO(boomanaiden154): Check the values of the validation counters to
  // ensure that the benchmarking runs are being run with the assumed
  // conditions.
  const std::unique_ptr<BenchmarkRunner> Runner =
      ExitOnErr(State.getExegesisTarget().createBenchmarkRunner(
          Benchmark::Latency, State, BenchmarkPhaseSelectorE::Measure,
          BenchmarkRunner::ExecutionModeE::SubProcess, 30, {}, Benchmark::Min));

  if (pfm::pfmInitialize())
    ExitOnErr(llvm::make_error<StringError>(inconvertibleErrorCode(),
                                            "Failed to initialize libpfm"));

  for (const auto &AnnotatedBlock : *ParsedAnnotatedBlocks.getAsArray()) {
    const llvm::json::Object *AnnotatedBlockObject =
        AnnotatedBlock.getAsObject();
    if (!AnnotatedBlockObject)
      ExitOnErr(llvm::make_error<StringError>(
          errc::invalid_argument,
          "Malformed basic block: is not a JSON object"));

    BenchmarkCode BenchCode = ExitOnErr(parseJSONBlock(
        *AnnotatedBlockObject, *MachinePrinter, *MachineDisassembler, State));

    std::unique_ptr<const SnippetRepetitor> SnipRepetitor =
        SnippetRepetitor::Create(Benchmark::RepetitionModeE::MiddleHalfLoop,
                                 State, BenchCode.Key.LoopRegister);

    // TODO(boomanaiden154): Refactor benchmark into a separate function?
    SmallVector<Benchmark, 2> AllResults;

    BenchmarkRunner::RunnableConfiguration RC1 = ExitOnErr(
        Runner->getRunnableConfiguration(BenchCode, 5000, 0, *SnipRepetitor));
    BenchmarkRunner::RunnableConfiguration RC2 = ExitOnErr(
        Runner->getRunnableConfiguration(BenchCode, 10000, 0, *SnipRepetitor));

    std::pair<Error, Benchmark> BenchmarkResult1OrErr =
        Runner->runConfiguration(std::move(RC1), {});

    if (std::get<0>(BenchmarkResult1OrErr)) {
      dbgs() << "Encountered an error while benchmarking: "
             << std::get<0>(BenchmarkResult1OrErr) << "\n";
      continue;
    }

    AllResults.push_back(std::move(std::get<1>(BenchmarkResult1OrErr)));

    std::pair<Error, Benchmark> BenchmarkResult2OrErr =
        Runner->runConfiguration(std::move(RC2), {});

    if (std::get<0>(BenchmarkResult2OrErr)) {
      dbgs() << "Encountered an error while benchmarking: "
             << std::get<0>(BenchmarkResult2OrErr) << "\n";
      continue;
    }

    AllResults.push_back(std::move(std::get<1>(BenchmarkResult2OrErr)));

    std::unique_ptr<ResultAggregator> ResultAgg =
        ResultAggregator::CreateAggregator(
            Benchmark::RepetitionModeE::MiddleHalfLoop);

    Benchmark Result = std::move(AllResults[0]);

    ResultAgg->AggregateResults(Result,
                                ArrayRef<Benchmark>(AllResults).drop_front());

    unsigned Throughput100 = static_cast<unsigned>(
        round(Result.Measurements[0].PerSnippetValue * 100));

    std::optional<StringRef> HexValue = AnnotatedBlockObject->getString("Hex");
    if (!HexValue)
      ExitOnErr(llvm::make_error<StringError>(
          errc::invalid_argument, "Expected basic block to have hex value"));

    outs() << *HexValue << "," << Throughput100 << "\n";
  }

  return 0;
}
