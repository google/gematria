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

#include "gematria/datasets/find_accessed_addrs_exegesis.h"

// Use the absolute path for headers from llvm-exegesis as there is no
// canonical include path within LLVM as they are not properly exposed through
// a library and could potentially be confused with other LLVM includes.

#include <unistd.h>

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86RegisterInfo.h"
#include "gematria/llvm/disassembler.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkRunner.h"
#include "llvm/tools/llvm-exegesis/lib/LlvmState.h"
#include "llvm/tools/llvm-exegesis/lib/Target.h"
#include "llvm/tools/llvm-exegesis/lib/TargetSelect.h"

using namespace llvm;
using namespace llvm::exegesis;

namespace gematria {

ExegesisAnnotator::ExegesisAnnotator(
    LLVMState &ExegesisState, std::unique_ptr<BenchmarkRunner> BenchRunner,
    std::unique_ptr<const SnippetRepetitor> SnipRepetitor)
    : State(ExegesisState),
      Runner(std::move(BenchRunner)),
      Repetitor(std::move(SnipRepetitor)) {
  MachineContext = std::make_unique<MCContext>(
      State.getTargetMachine().getTargetTriple(),
      State.getTargetMachine().getMCAsmInfo(), &State.getRegInfo(),
      &State.getSubtargetInfo());
  MachineDisassembler.reset(
      State.getTargetMachine().getTarget().createMCDisassembler(
          State.getSubtargetInfo(), *MachineContext));
  MachinePrinter.reset(State.getTargetMachine().getTarget().createMCInstPrinter(
      State.getTargetMachine().getTargetTriple(), 0,
      *State.getTargetMachine().getMCAsmInfo(), State.getInstrInfo(),
      State.getRegInfo()));
}

Expected<std::unique_ptr<ExegesisAnnotator>> ExegesisAnnotator::create(
    LLVMState &ExegesisState) {
  // Initialize the supported Exegesis targets. Currently we only support X86.
  InitializeX86ExegesisTarget();

  if (pfm::pfmInitialize())
    return make_error<StringError>(
        "Failed to initialize libpfm",
        std::make_error_code(std::errc::invalid_argument));

  auto RunnerOrErr = ExegesisState.getExegesisTarget().createBenchmarkRunner(
      Benchmark::Latency, ExegesisState, BenchmarkPhaseSelectorE::Measure,
      BenchmarkRunner::ExecutionModeE::SubProcess, 1, {}, Benchmark::Min);

  if (!RunnerOrErr) return RunnerOrErr.takeError();

  std::unique_ptr<const SnippetRepetitor> SnipRepetitor =
      SnippetRepetitor::Create(Benchmark::RepetitionModeE::Duplicate,
                               ExegesisState, X86::R8);

  return std::unique_ptr<ExegesisAnnotator>(new ExegesisAnnotator(
      ExegesisState, std::move(*RunnerOrErr), std::move(SnipRepetitor)));
}

Expected<AccessedAddrs> ExegesisAnnotator::findAccessedAddrs(
    ArrayRef<uint8_t> BasicBlock) {
  Expected<std::vector<DisassembledInstruction>> DisInstructions =
      DisassembleAllInstructions(*MachineDisassembler, State.getInstrInfo(),
                                 State.getRegInfo(), State.getSubtargetInfo(),
                                 *MachinePrinter, 0, BasicBlock);

  if (!DisInstructions) return DisInstructions.takeError();

  std::vector<MCInst> Instructions;
  Instructions.reserve(DisInstructions->size());

  for (const auto &DisInstruction : *DisInstructions)
    Instructions.push_back(DisInstruction.mc_inst);

  AccessedAddrs MemAnnotations;
  MemAnnotations.code_location = 0;
  MemAnnotations.block_size = 4096;

  BenchmarkCode BenchCode;
  BenchCode.Key.Instructions = Instructions;

  MemoryValue MemVal;
  MemVal.Value = APInt(64, 0x12345600);
  MemVal.Index = 0;
  MemVal.SizeBytes = 4096;

  BenchCode.Key.MemoryValues["memdef1"] = MemVal;

  const llvm::MCRegisterInfo &MRI = State.getRegInfo();

  for (unsigned i = 0;
       i < MRI.getRegClass(X86::GR64_NOREX2RegClassID).getNumRegs(); ++i) {
    RegisterValue RegVal;
    RegVal.Register =
        MRI.getRegClass(X86::GR64_NOREX2RegClassID).getRegister(i);
    RegVal.Value = APInt(64, 0x12345600);
    BenchCode.Key.RegisterInitialValues.push_back(RegVal);
  }

  for (unsigned i = 0; i < MRI.getRegClass(X86::VR128RegClassID).getNumRegs();
       ++i) {
    RegisterValue RegVal;
    RegVal.Register = MRI.getRegClass(X86::VR128RegClassID).getRegister(i);
    RegVal.Value = APInt(128, 0x12345600);
    BenchCode.Key.RegisterInitialValues.push_back(RegVal);
  }

  while (true) {
    std::unique_ptr<const SnippetRepetitor> SR = SnippetRepetitor::Create(
        Benchmark::RepetitionModeE::Duplicate, State, X86::R8);
    Expected<BenchmarkRunner::RunnableConfiguration> RCOrErr =
        Runner->getRunnableConfiguration(BenchCode, 10000, 0, *SR);

    if (!RCOrErr) return RCOrErr.takeError();

    BenchmarkRunner::RunnableConfiguration &RC = *RCOrErr;

    std::pair<Error, Benchmark> BenchmarkResultOrErr =
        Runner->runConfiguration(std::move(RC), {});

    // If we don't have any errors executing the snippet, we executed the
    // snippet successfully and thus have all the needed memory annotations.
    if (!std::get<0>(BenchmarkResultOrErr)) break;

    if (!std::get<0>(BenchmarkResultOrErr).isA<SnippetSegmentationFault>())
      return std::move(std::get<0>(BenchmarkResultOrErr));

    Error AnnotationError = handleErrors(
        std::move(std::get<0>(BenchmarkResultOrErr)),
        [&](SnippetSegmentationFault &CrashInfo) -> Error {
          MemoryMapping MemMap;
          // Zero out the last twelve bits of the address to align
          // the address to a page boundary.
          uintptr_t MapAddress =
              (CrashInfo.getAddress() / getpagesize()) * getpagesize();
          if (MapAddress == 0)
            return make_error<Failure>("Segfault at zero address, cannot map.");
          // TODO(boomanaiden154): The fault captured below occurs when
          // exegesis tries to map an address and the mmap fails. When these
          // errors are handled within exegesis, we should remove this check.
          for (const auto &MemoryMapping : BenchCode.Key.MemoryMappings) {
            if (MemoryMapping.Address == MapAddress)
              return make_error<Failure>(
                  "Segfault at an address already mapped.");
          }
          MemMap.Address = MapAddress;
          MemMap.MemoryValueName = "memdef1";
          BenchCode.Key.MemoryMappings.push_back(MemMap);

          return Error::success();
        });

    if (AnnotationError) return std::move(AnnotationError);
  }

  MemAnnotations.accessed_blocks.reserve(BenchCode.Key.MemoryMappings.size());

  for (const MemoryMapping &Mapping : BenchCode.Key.MemoryMappings) {
    MemAnnotations.accessed_blocks.push_back(Mapping.Address);
  }

  return MemAnnotations;
}

}  // namespace gematria
