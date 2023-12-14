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

#include "BenchmarkRunner.h"
#include "LlvmState.h"
#include "Target.h"
#include "TargetSelect.h"
#include "X86.h"
#include "X86InstrInfo.h"
#include "X86RegisterInfo.h"
#include "gematria/llvm/disassembler.h"

using namespace llvm;
using namespace llvm::exegesis;

namespace gematria {

ExegesisAnnotator::ExegesisAnnotator(
    LlvmArchitectureSupport &ArchSupport_, LLVMState &State_,
    std::unique_ptr<BenchmarkRunner> Runner_,
    std::unique_ptr<const SnippetRepetitor> Repetitor_)
    : ArchSupport(ArchSupport_),
      MCPrinter(ArchSupport.CreateMCInstPrinter(0)),
      State(State_),
      Runner(std::move(Runner_)),
      Repetitor(std::move(Repetitor_)) {}

Expected<std::unique_ptr<ExegesisAnnotator>> ExegesisAnnotator::Create(
    LlvmArchitectureSupport &ArchSupport_, LLVMState &State_) {
  // Initialize the supported Exegesis targets. Currently we only support X86.
  InitializeX86ExegesisTarget();

  if (pfm::pfmInitialize())
    return make_error<StringError>(
        "Failed to initialize libpfm",
        std::make_error_code(std::errc::invalid_argument));

  auto RunnerOrErr = State_.getExegesisTarget().createBenchmarkRunner(
      Benchmark::Latency, State_, BenchmarkPhaseSelectorE::Measure,
      BenchmarkRunner::ExecutionModeE::SubProcess, 1, Benchmark::Min);

  if (!RunnerOrErr) return RunnerOrErr.takeError();

  std::unique_ptr<const SnippetRepetitor> Repetitor_ =
      SnippetRepetitor::Create(Benchmark::RepetitionModeE::Duplicate, State_);

  return std::unique_ptr<ExegesisAnnotator>(new ExegesisAnnotator(
      ArchSupport_, State_, std::move(*RunnerOrErr), std::move(Repetitor_)));
}

Expected<AccessedAddrs> ExegesisAnnotator::FindAccessedAddrs(
    ArrayRef<uint8_t> BasicBlock) {
  Expected<std::vector<DisassembledInstruction>> DisInstructions =
      DisassembleAllInstructions(
          ArchSupport.mc_disassembler(), ArchSupport.mc_instr_info(),
          ArchSupport.mc_register_info(), ArchSupport.mc_subtarget_info(),
          *MCPrinter, 0, BasicBlock);

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

  const llvm::MCRegisterInfo &MRI = ArchSupport.mc_register_info();

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
    std::unique_ptr<const SnippetRepetitor> SR =
        SnippetRepetitor::Create(Benchmark::RepetitionModeE::Duplicate, State);
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

    handleAllErrors(std::move(std::get<0>(BenchmarkResultOrErr)),
                    [&](SnippetSegmentationFault &CrashInfo) {
                      MemoryMapping MemMap;
                      MemMap.Address = CrashInfo.getAddress();
                      MemMap.MemoryValueName = "memdef1";
                      BenchCode.Key.MemoryMappings.push_back(MemMap);
                    });
  }

  MemAnnotations.accessed_blocks.reserve(BenchCode.Key.MemoryMappings.size());

  for (const MemoryMapping &Mapping : BenchCode.Key.MemoryMappings) {
    MemAnnotations.accessed_blocks.push_back(Mapping.Address);
  }

  return MemAnnotations;
}

}  // namespace gematria
