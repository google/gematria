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
    const MCDisassembler &MachineDisassembler, const LLVMState &State,
    size_t BlockIndex) {
  BenchmarkCode BenchCode;

  std::optional<StringRef> HexValue = BasicBlockJSON.getString("Hex");
  if (!HexValue)
    return llvm::make_error<StringError>(
        errc::invalid_argument, "Malformed basic block: Basic block at index " +
                                    Twine(BlockIndex) + " has no hex value");

  std::optional<int64_t> LoopRegister =
      BasicBlockJSON.getInteger("LoopRegister");
  if (!LoopRegister.has_value())
    return llvm::make_error<StringError>(
        errc::invalid_argument, "Malformed basic block: Basic block at index " +
                                    Twine(BlockIndex) +
                                    " has no loop register");

  BenchCode.Key.LoopRegister = *LoopRegister;

  // TODO(ondrasej): Update this after converting gematria::ParseHexString to
  // return llvm::Expected rather than an optional.
  std::optional<std::vector<uint8_t>> BytesOr =
      gematria::ParseHexString(HexValue->str());

  if (!BytesOr.has_value())
    return llvm::make_error<StringError>(
        errc::invalid_argument, "Malformed basic block: Basic block at index " +
                                    Twine(BlockIndex) +
                                    " has an invalid hex value: " + *HexValue);

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
        errc::invalid_argument, "Malformed basic block: Basic block at index " +
                                    Twine(BlockIndex) +
                                    " has no register definitions array");

  size_t RegisterLoopIndex = 0;
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
          "Malformed register definition: Basic block at index " +
              Twine(BlockIndex) +
              " is missing a register number or value for register at index " +
              Twine(RegisterLoopIndex));

    RegVal.Register = *RegisterIndex;
    RegVal.Value = APInt(64, *RegisterValue);
    BenchCode.Key.RegisterInitialValues.push_back(RegVal);

    ++RegisterLoopIndex;
  }

  const json::Array *MemoryDefinitions =
      BasicBlockJSON.getArray("MemoryDefinitions");

  if (!MemoryDefinitions)
    return llvm::make_error<StringError>(
        errc::invalid_argument, "Malformed basic block: Basic block at index " +
                                    Twine(BlockIndex) +
                                    " has no memory definitions array");

  size_t MemoryDefinitionIndex = 0;

  for (const auto &MemoryDefinitionValue : *MemoryDefinitions) {
    const json::Object *MemoryDefinitionObject =
        MemoryDefinitionValue.getAsObject();

    if (!MemoryDefinitionObject)
      return llvm::make_error<StringError>(
          errc::invalid_argument,
          "Malformed memory definition: Basic block at index " +
              Twine(BlockIndex) + " has a memory definition at index " +
              Twine(MemoryDefinitionIndex) + " that is not a JSON object");

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
          "Malformed memory definition: Basic block at index " +
              Twine(BlockIndex) + " has a memory definition at index " +
              Twine(MemoryDefinitionIndex) + " with no size, name, or value");

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
        errc::invalid_argument, "Malformed basic block: Basic block at index " +
                                    Twine(BlockIndex) +
                                    " has no memory mappings array");

  for (size_t I = 0; I < MemoryMappings->size(); ++I) {
    const json::Object *MemoryMappingObject =
        MemoryMappings->data()[I].getAsObject();

    if (!MemoryMappingObject)
      return llvm::make_error<StringError>(
          errc::invalid_argument,
          "Malformed memory mapping: Basic block at index " +
              Twine(BlockIndex) + " has a memory mapping at index " + Twine(I) +
              " which is not a JSON object");

    std::optional<StringRef> MemoryMappingDefinitionName =
        MemoryMappingObject->getString("Value");
    std::optional<uintptr_t> MemoryMappingAddress =
        MemoryMappingObject->getInteger("Address");

    if (!MemoryMappingDefinitionName.has_value() ||
        !MemoryMappingAddress.has_value())
      return llvm::make_error<StringError>(
          errc::invalid_argument,
          "Malformed memory mapping: Basic block at index " +
              Twine(BlockIndex) + " has a memory mapping at index " + Twine(I) +
              " with no name or address");

    MemoryMapping MemMap;
    MemMap.Address = *MemoryMappingAddress;
    MemMap.MemoryValueName = MemoryMappingDefinitionName->str();
    BenchCode.Key.MemoryMappings.push_back(MemMap);
  }

  return BenchCode;
}

Expected<double> benchmarkBasicBlock(const BenchmarkCode &BenchCode,
                                     const BenchmarkRunner &BenchRunner,
                                     const LLVMState &State) {
  std::unique_ptr<const SnippetRepetitor> SnipRepetitor =
      SnippetRepetitor::Create(Benchmark::RepetitionModeE::MiddleHalfLoop,
                               State, BenchCode.Key.LoopRegister);

  SmallVector<Benchmark, 2> AllResults;

  auto RC1 =
      BenchRunner.getRunnableConfiguration(BenchCode, 5000, 0, *SnipRepetitor);
  if (!RC1) return RC1.takeError();
  auto RC2 =
      BenchRunner.getRunnableConfiguration(BenchCode, 10000, 0, *SnipRepetitor);
  if (!RC2) return RC2.takeError();

  std::pair<Error, Benchmark> BenchmarkResultAOrErr =
      BenchRunner.runConfiguration(std::move(*RC1), {});

  if (std::get<0>(BenchmarkResultAOrErr))
    return std::move(std::get<0>(BenchmarkResultAOrErr));

  AllResults.push_back(std::move(std::get<1>(BenchmarkResultAOrErr)));

  std::pair<Error, Benchmark> BenchmarkResultBOrErr =
      BenchRunner.runConfiguration(std::move(*RC2), {});

  if (std::get<0>(BenchmarkResultBOrErr))
    return std::move(std::get<0>(BenchmarkResultBOrErr));

  AllResults.push_back(std::move(std::get<1>(BenchmarkResultBOrErr)));

  std::unique_ptr<ResultAggregator> ResultAgg =
      ResultAggregator::CreateAggregator(
          Benchmark::RepetitionModeE::MiddleHalfLoop);

  Benchmark Result = std::move(AllResults[0]);

  ResultAgg->AggregateResults(Result,
                              ArrayRef<Benchmark>(AllResults).drop_front());

  return Result.Measurements[0].PerSnippetValue;
}

int main(int Argc, char *Argv[]) {
  cl::ParseCommandLineOptions(
      Argc, Argv, "Tool for benchmarking sets of annotated basic blocks");

  ExitOnError ExitOnErr("exegesis-benchmark error: ");
  if (AnnotatedBlocksJson.empty())
    ExitOnErr(llvm::make_error<StringError>(
        errc::invalid_argument, "--annotated_blocks_json is required"));

  // LLVM Setup
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86Target();
  LLVMInitializeX86AsmPrinter();
  LLVMInitializeX86AsmParser();
  LLVMInitializeX86Disassembler();

  // Exegesis Setup
  InitializeX86ExegesisTarget();

  if (pfm::pfmInitialize())
    ExitOnErr(llvm::make_error<StringError>(inconvertibleErrorCode(),
                                            "Failed to initialize libpfm"));

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

  auto JsonMemoryBuffer = ExitOnErr(
      errorOrToExpected(MemoryBuffer::getFile(AnnotatedBlocksJson, true)));

  auto ParsedAnnotatedBlocks =
      ExitOnErr(json::parse(JsonMemoryBuffer->getBuffer()));

  size_t BlockIndex = 0;
  for (const auto &AnnotatedBlock : *ParsedAnnotatedBlocks.getAsArray()) {
    const llvm::json::Object *AnnotatedBlockObject =
        AnnotatedBlock.getAsObject();
    if (!AnnotatedBlockObject)
      ExitOnErr(llvm::make_error<StringError>(
          errc::invalid_argument,
          "Malformed basic block: is not a JSON object"));

    Expected<BenchmarkCode> BenchCodeOrErr =
        parseJSONBlock(*AnnotatedBlockObject, *MachinePrinter,
                       *MachineDisassembler, State, BlockIndex);
    if (!BenchCodeOrErr)
      ExitOnErr(
          createFileError(AnnotatedBlocksJson, BenchCodeOrErr.takeError()));

    const BenchmarkCode &BenchCode = *BenchCodeOrErr;

    std::unique_ptr<const SnippetRepetitor> SnipRepetitor =
        SnippetRepetitor::Create(Benchmark::RepetitionModeE::MiddleHalfLoop,
                                 State, BenchCode.Key.LoopRegister);

    Expected<double> ThroughputOrErr =
        benchmarkBasicBlock(BenchCode, *Runner, State);

    if (!ThroughputOrErr)
      ExitOnErr(
          createFileError(AnnotatedBlocksJson, ThroughputOrErr.takeError()));

    std::optional<StringRef> HexValue = AnnotatedBlockObject->getString("Hex");
    // The block has already been parsed previously, and thus should have thrown
    // an error if there is no hex value. Assert that this is the case here.
    assert(HexValue.has_value() &&
           "Expected block to already have been checked for a hex value.");

    outs() << *HexValue << "," << *ThroughputOrErr << "\n";

    ++BlockIndex;
  }

  return 0;
}
