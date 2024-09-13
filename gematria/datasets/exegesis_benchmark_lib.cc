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

#include "gematria/datasets/exegesis_benchmark_lib.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "gematria/llvm/disassembler.h"
#include "gematria/proto/execution_annotation.pb.h"
#include "gematria/utils/string.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkCode.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkResult.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkRunner.h"
#include "llvm/tools/llvm-exegesis/lib/LlvmState.h"
#include "llvm/tools/llvm-exegesis/lib/RegisterValue.h"
#include "llvm/tools/llvm-exegesis/lib/ResultAggregator.h"
#include "llvm/tools/llvm-exegesis/lib/SnippetRepetitor.h"
#include "llvm/tools/llvm-exegesis/lib/Target.h"

using namespace llvm;
using namespace llvm::exegesis;

namespace gematria {

ExegesisBenchmark::ExegesisBenchmark(LLVMState &&State)
    : ExegesisState(std::move(State)) {
  LLVMMCContext = std::make_unique<MCContext>(
      ExegesisState.getTargetMachine().getTargetTriple(),
      ExegesisState.getTargetMachine().getMCAsmInfo(),
      &ExegesisState.getRegInfo(), &ExegesisState.getSubtargetInfo());

  LLVMMCDisassembler = std::unique_ptr<MCDisassembler>(
      ExegesisState.getTargetMachine().getTarget().createMCDisassembler(
          ExegesisState.getSubtargetInfo(), *LLVMMCContext));

  LLVMMCInstPrinter = std::unique_ptr<MCInstPrinter>(
      ExegesisState.getTargetMachine().getTarget().createMCInstPrinter(
          ExegesisState.getTargetMachine().getTargetTriple(),
          llvm::InlineAsm::AD_ATT,
          *ExegesisState.getTargetMachine().getMCAsmInfo(),
          *ExegesisState.getTargetMachine().getMCInstrInfo(),
          ExegesisState.getRegInfo()));
}

Expected<std::unique_ptr<ExegesisBenchmark>> ExegesisBenchmark::create() {
  Expected<LLVMState> StateOrErr = LLVMState::Create("", "native");

  if (!StateOrErr) return StateOrErr.takeError();

  std::unique_ptr<ExegesisBenchmark> benchmark_instance(
      new ExegesisBenchmark(std::move(*StateOrErr)));

  // TODO(boomanaiden154): Check the values of the validation counters to
  // ensure that the benchmarking runs are being run with the assumed
  // conditions.
  Expected<std::unique_ptr<BenchmarkRunner>> RunnerOrErr =
      benchmark_instance->ExegesisState.getExegesisTarget()
          .createBenchmarkRunner(Benchmark::Latency,
                                 benchmark_instance->ExegesisState,
                                 BenchmarkPhaseSelectorE::Measure,
                                 BenchmarkRunner::ExecutionModeE::SubProcess,
                                 30, {}, Benchmark::Min);

  if (!RunnerOrErr) return RunnerOrErr.takeError();

  benchmark_instance->BenchRunner = std::move(*RunnerOrErr);

  return benchmark_instance;
}

Expected<BenchmarkCode> ExegesisBenchmark::parseJSONBlock(
    const json::Object &BasicBlockJSON, size_t BlockIndex) {
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
  std::optional<std::vector<uint8_t>> BytesOr = ParseHexString(HexValue->str());

  if (!BytesOr.has_value())
    return llvm::make_error<StringError>(
        errc::invalid_argument, "Malformed basic block: Basic block at index " +
                                    Twine(BlockIndex) +
                                    " has an invalid hex value: " + *HexValue);

  Expected<std::vector<DisassembledInstruction>> DisInstructionsOrErr =
      DisassembleAllInstructions(
          *LLVMMCDisassembler, ExegesisState.getInstrInfo(),
          ExegesisState.getRegInfo(), ExegesisState.getSubtargetInfo(),
          *LLVMMCInstPrinter, 0, *BytesOr);

  if (!DisInstructionsOrErr) return DisInstructionsOrErr.takeError();

  std::vector<MCInst> Instructions;
  Instructions.reserve(DisInstructionsOrErr->size());

  for (const auto &DisInstruction : *DisInstructionsOrErr)
    Instructions.push_back(DisInstruction.mc_inst);

  BenchCode.Key.Instructions = std::move(Instructions);

  const json::Array *RegisterDefinitions =
      BasicBlockJSON.getArray("RegisterDefinitions");

  Twine BasicBlockAtIndex = "Basic block at index " + Twine(BlockIndex);

  if (!RegisterDefinitions)
    return llvm::make_error<StringError>(
        errc::invalid_argument, "Malformed basic block: " + BasicBlockAtIndex +
                                    " has no register definitions array");

  for (size_t RegisterLoopIndex = 0;
       RegisterLoopIndex < RegisterDefinitions->size(); ++RegisterLoopIndex) {
    const json::Object *RegisterDefinitionObject =
        (*RegisterDefinitions)[RegisterLoopIndex].getAsObject();

    RegisterValue RegVal;
    std::optional<StringRef> RegisterName =
        RegisterDefinitionObject->getString("Register");
    std::optional<int64_t> RegisterValue =
        RegisterDefinitionObject->getInteger("Value");
    if (!RegisterName.has_value() || !RegisterValue.has_value())
      return llvm::make_error<StringError>(
          errc::invalid_argument,
          "Malformed register definition: " + BasicBlockAtIndex +
              " is missing a register number or value for register at index " +
              Twine(RegisterLoopIndex));

    Expected<MCRegister> RegisterIndex = getRegisterFromName(*RegisterName);
    if (!RegisterIndex) return RegisterIndex.takeError();

    RegVal.Register = *RegisterIndex;
    RegVal.Value = APInt(64, *RegisterValue);
    BenchCode.Key.RegisterInitialValues.push_back(RegVal);
  }

  const json::Array *MemoryDefinitions =
      BasicBlockJSON.getArray("MemoryDefinitions");

  if (!MemoryDefinitions)
    return llvm::make_error<StringError>(
        errc::invalid_argument, "Malformed basic block: " + BasicBlockAtIndex +
                                    " has no memory definitions array");

  size_t MemoryDefinitionIndex = 0;

  for (const auto &MemoryDefinitionValue : *MemoryDefinitions) {
    const json::Object *MemoryDefinitionObject =
        MemoryDefinitionValue.getAsObject();

    if (!MemoryDefinitionObject)
      return llvm::make_error<StringError>(
          errc::invalid_argument,
          "Malformed memory definition: " + BasicBlockAtIndex +
              " has a memory definition at index " +
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
          "Malformed memory definition: " + BasicBlockAtIndex +
              " has a memory definition at index " +
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
        errc::invalid_argument, "Malformed basic block: " + BasicBlockAtIndex +
                                    " has no memory mappings array");

  for (size_t I = 0; I < MemoryMappings->size(); ++I) {
    const json::Object *MemoryMappingObject =
        MemoryMappings->data()[I].getAsObject();

    if (!MemoryMappingObject)
      return llvm::make_error<StringError>(
          errc::invalid_argument,
          "Malformed memory mapping: " + BasicBlockAtIndex +
              " has a memory mapping at index " + Twine(I) +
              " which is not a JSON object");

    std::optional<StringRef> MemoryMappingDefinitionName =
        MemoryMappingObject->getString("Value");
    std::optional<uintptr_t> MemoryMappingAddress =
        MemoryMappingObject->getInteger("Address");

    if (!MemoryMappingDefinitionName.has_value() ||
        !MemoryMappingAddress.has_value())
      return llvm::make_error<StringError>(
          errc::invalid_argument,
          "Malformed memory mapping: " + BasicBlockAtIndex +
              " has a memory mapping at index " + Twine(I) +
              " with no name or address");

    MemoryMapping MemMap;
    MemMap.Address = *MemoryMappingAddress;
    MemMap.MemoryValueName = MemoryMappingDefinitionName->str();
    BenchCode.Key.MemoryMappings.push_back(MemMap);
  }

  return BenchCode;
}

Expected<BenchmarkCode> ExegesisBenchmark::processAnnotatedBlock(
    std::string_view BlockHex, const ExecutionAnnotations &Annotations) {
  std::optional<std::vector<uint8_t>> Bytes =
      gematria::ParseHexString(BlockHex);

  if (!Bytes.has_value())
    return llvm::make_error<StringError>(
        errc::invalid_argument,
        "Malformed basic block: invalid hex value " + BlockHex);

  Expected<std::vector<gematria::DisassembledInstruction>>
      DisassembledInstructions = gematria::DisassembleAllInstructions(
          *LLVMMCDisassembler, ExegesisState.getInstrInfo(),
          ExegesisState.getRegInfo(), ExegesisState.getSubtargetInfo(),
          *LLVMMCInstPrinter, 0, *Bytes);

  if (!DisassembledInstructions) return DisassembledInstructions.takeError();

  BenchmarkCode BenchmarkConfiguration;

  std::vector<MCInst> Instructions;
  Instructions.reserve(DisassembledInstructions->size());

  for (const gematria::DisassembledInstruction &Instruction :
       *DisassembledInstructions)
    Instructions.push_back(Instruction.mc_inst);

  BenchmarkConfiguration.Key.Instructions = std::move(Instructions);

  BenchmarkConfiguration.Key.RegisterInitialValues.reserve(
      Annotations.initial_registers_size());

  for (const RegisterAndValue &RegisterValue :
       Annotations.initial_registers()) {
    llvm::Expected<MCRegister> RegisterIndex =
        getRegisterFromName(RegisterValue.register_name());
    if (!RegisterIndex) return RegisterIndex.takeError();

    struct RegisterValue ValueToAdd = {
        .Register = *RegisterIndex,
        .Value = APInt(64, RegisterValue.register_value())};

    BenchmarkConfiguration.Key.RegisterInitialValues.push_back(
        std::move(ValueToAdd));
  }

  MemoryValue MemVal = {.Value = APInt(64, Annotations.block_contents()),
                        .SizeBytes = Annotations.block_size(),
                        .Index = 0};
  BenchmarkConfiguration.Key.MemoryValues["MEM"] = std::move(MemVal);

  BenchmarkConfiguration.Key.MemoryMappings.reserve(
      Annotations.accessed_blocks_size());

  for (const uintptr_t AccessedBlock : Annotations.accessed_blocks()) {
    MemoryMapping MemMap = {.Address = AccessedBlock, .MemoryValueName = "MEM"};

    BenchmarkConfiguration.Key.MemoryMappings.push_back(std::move(MemMap));
  }

  BenchmarkConfiguration.Key.SnippetAddress = Annotations.code_start_address();

  if (Annotations.has_loop_register()) {
    Expected<MCRegister> LoopRegisterIndex =
        getRegisterFromName(Annotations.loop_register());
    if (!LoopRegisterIndex) return LoopRegisterIndex.takeError();
    BenchmarkConfiguration.Key.LoopRegister = *LoopRegisterIndex;
  } else {
    BenchmarkConfiguration.Key.LoopRegister = MCRegister::NoRegister;
  }

  return BenchmarkConfiguration;
}

Expected<double> ExegesisBenchmark::benchmarkBasicBlock(
    const BenchmarkCode &BenchCode) {
  std::unique_ptr<const SnippetRepetitor> SnipRepetitor =
      SnippetRepetitor::Create(Benchmark::RepetitionModeE::MiddleHalfLoop,
                               ExegesisState, BenchCode.Key.LoopRegister);
  SmallVector<Benchmark, 2> AllResults;

  auto RC1 =
      BenchRunner->getRunnableConfiguration(BenchCode, 5000, 0, *SnipRepetitor);
  if (!RC1) return RC1.takeError();
  auto RC2 = BenchRunner->getRunnableConfiguration(BenchCode, 10000, 0,
                                                   *SnipRepetitor);
  if (!RC2) return RC2.takeError();

  std::pair<Error, Benchmark> BenchmarkResultAOrErr =
      BenchRunner->runConfiguration(std::move(*RC1), {});

  if (std::get<0>(BenchmarkResultAOrErr))
    return std::move(std::get<0>(BenchmarkResultAOrErr));

  AllResults.push_back(std::move(std::get<1>(BenchmarkResultAOrErr)));

  std::pair<Error, Benchmark> BenchmarkResultBOrErr =
      BenchRunner->runConfiguration(std::move(*RC2), {});

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

Expected<MCRegister> ExegesisBenchmark::getRegisterFromName(
    StringRef RegisterName) {
  auto RegIterator =
      ExegesisState.getRegNameToRegNoMapping().find(RegisterName);
  if (RegIterator == ExegesisState.getRegNameToRegNoMapping().end())
    return llvm::make_error<StringError>(
        errc::invalid_argument,
        "Invalid register name for target: " + RegisterName);
  return RegIterator->second;
}

}  // namespace gematria
