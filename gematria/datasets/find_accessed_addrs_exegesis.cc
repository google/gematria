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

#include <cstdint>
#include <memory>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

#include "gematria/datasets/basic_block_utils.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/proto/execution_annotation.pb.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/bit.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/lib/Target/X86/MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkCode.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkResult.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkRunner.h"
#include "llvm/tools/llvm-exegesis/lib/Error.h"
#include "llvm/tools/llvm-exegesis/lib/LlvmState.h"
#include "llvm/tools/llvm-exegesis/lib/PerfHelper.h"
#include "llvm/tools/llvm-exegesis/lib/RegisterValue.h"
#include "llvm/tools/llvm-exegesis/lib/SnippetRepetitor.h"
#include "llvm/tools/llvm-exegesis/lib/Target.h"
#include "llvm/tools/llvm-exegesis/lib/TargetSelect.h"

using namespace llvm;
using namespace llvm::exegesis;

// Use the constants from the BHive paper for setting initial register and
// memory values. These constants are set to a high enough value to avoid
// underflow and accesses within the first page, but low enough to avoid
// exceeding the virtual address space ceiling in most cases.
constexpr uint64_t kInitialRegVal = 0x12345600;
constexpr uint64_t kInitialMemVal = 0x12345600;

// TODO(boomanaiden154): This is currently chosen just to make things simple.
// Eventually we should probably switch to a different constant per each FP
// register size (rather than just enumerating floats/doubles) and ensure it
// is a reasonable enough value to hopefully avoid problems like expensive FP
// operations due to edge cases. These values should be fine, but more
// experimentation is needed.
constexpr float kInitialFloatVal = 1000000.0f;
constexpr double kInitialDoubleVal = 1000000.0f;

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

Expected<ExecutionAnnotations> ExegesisAnnotator::findAccessedAddrs(
    ArrayRef<uint8_t> BasicBlock, unsigned MaxAnnotationAttempts) {
  Expected<std::vector<DisassembledInstruction>> DisInstructions =
      DisassembleAllInstructions(*MachineDisassembler, State.getInstrInfo(),
                                 State.getRegInfo(), State.getSubtargetInfo(),
                                 *MachinePrinter, 0, BasicBlock);

  if (!DisInstructions) return DisInstructions.takeError();

  std::vector<MCInst> Instructions;
  Instructions.reserve(DisInstructions->size());

  for (const auto &DisInstruction : *DisInstructions)
    Instructions.push_back(DisInstruction.mc_inst);

  ExecutionAnnotations MemAnnotations;
  MemAnnotations.set_code_start_address(0);
  MemAnnotations.set_block_size(4096);

  BenchmarkCode BenchCode;
  BenchCode.Key.Instructions = Instructions;

  MemoryValue MemVal;
  MemVal.Value = APInt(64, kInitialMemVal);
  MemVal.Index = 0;
  MemVal.SizeBytes = 4096;

  BenchCode.Key.MemoryValues["memdef1"] = MemVal;

  const llvm::MCRegisterInfo &MRI = State.getRegInfo();
  const std::vector<unsigned> UsedRegisters = gematria::getUsedRegisters(
      *DisInstructions, State.getRegInfo(), State.getInstrInfo());
  MemAnnotations.mutable_accessed_blocks()->Reserve(
      BenchCode.Key.MemoryMappings.size());

  for (unsigned RegisterIndex : UsedRegisters) {
    // Skip all the segment registers other than FS and GS as we cannot set
    // them from userspace and the machine code verifier does not complain if
    // they are not defined.
    if (RegisterIndex == X86::SS || RegisterIndex == X86::CS ||
        RegisterIndex == X86::DS || RegisterIndex == X86::ES) {
      continue;
    }

    // We cannot explicitly set FPSW, so skip it if it comes up.
    if (RegisterIndex == X86::FPSW) {
      continue;
    }

    RegisterAndValue *NewRegisterValue = MemAnnotations.add_initial_registers();
    NewRegisterValue->set_register_name(
        State.getRegInfo().getName(RegisterIndex));

    RegisterValue RegVal;
    RegVal.Register = RegisterIndex;

    if (MRI.getRegClass(X86::GR64_NOREX2RegClassID).contains(RegisterIndex) ||
        MRI.getRegClass(X86::VR64RegClassID).contains(RegisterIndex) ||
        MRI.getRegClass(X86::VR128XRegClassID).contains(RegisterIndex) ||
        MRI.getRegClass(X86::VR256XRegClassID).contains(RegisterIndex) ||
        MRI.getRegClass(X86::VR512RegClassID).contains(RegisterIndex) ||
        MRI.getRegClass(X86::VK64RegClassID).contains(RegisterIndex)) {
      RegVal.Value = APInt(64, kInitialRegVal);
      NewRegisterValue->set_register_value(kInitialRegVal);
    } else if (RegisterIndex == X86::FS || RegisterIndex == X86::GS) {
      RegVal.Value = APInt(32, kInitialRegVal);
      NewRegisterValue->set_register_value(kInitialRegVal);
    } else if (RegisterIndex == X86::EFLAGS || RegisterIndex == X86::MXCSR ||
               RegisterIndex == X86::FPCW || RegisterIndex == X86::DF) {
      RegVal.Value = APInt(32, 0);
      NewRegisterValue->set_register_value(0);
    } else if (MRI.getRegClass(X86::RFP32RegClassID).contains(RegisterIndex)) {
      RegVal.Value = APInt(32, bit_cast<uint32_t>(kInitialFloatVal));
      NewRegisterValue->set_register_value(
          bit_cast<uint32_t>(kInitialFloatVal));
    } else if (MRI.getRegClass(X86::RSTRegClassID).contains(RegisterIndex) ||
               MRI.getRegClass(X86::RFP64RegClassID).contains(RegisterIndex) ||
               MRI.getRegClass(X86::RFP80RegClassID).contains(RegisterIndex)) {
      RegVal.Value = APInt(64, bit_cast<uint64_t>(kInitialDoubleVal));
      NewRegisterValue->set_register_value(
          bit_cast<uint64_t>(kInitialDoubleVal));
    } else {
      report_fatal_error(
          formatv("Expected all registers to be handled, but found unhandled "
                  "register {0}",
                  MRI.getName(RegisterIndex)));
    }

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
        Runner->runConfiguration(std::move(RC), {}, std::nullopt);

    // If we don't have any errors executing the snippet, we executed the
    // snippet successfully and thus have all the needed memory annotations.
    if (!std::get<0>(BenchmarkResultOrErr)) break;

    // If we run into an error where the perf counter has not been enabled the
    // entire time, we can ignore it. This error can only occur after the entire
    // snippet has executed and we do not care about performance measurements
    // in the annotation stage.
    if (std::get<0>(BenchmarkResultOrErr).isA<PerfCounterNotFullyEnabled>())
      break;

    if (!std::get<0>(BenchmarkResultOrErr).isA<SnippetSegmentationFault>())
      return std::move(std::get<0>(BenchmarkResultOrErr));

    Error AnnotationError = handleErrors(
        std::move(std::get<0>(BenchmarkResultOrErr)),
        [&](SnippetSegmentationFault &CrashInfo) -> Error {
          if (BenchCode.Key.MemoryMappings.size() > MaxAnnotationAttempts)
            return make_error<Failure>(
                "Hit the maximum number of annotation attempts.");

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

  for (const MemoryMapping &Mapping : BenchCode.Key.MemoryMappings) {
    MemAnnotations.add_accessed_blocks(Mapping.Address);
  }

  MemAnnotations.mutable_initial_registers()->Reserve(UsedRegisters.size());

  std::optional<unsigned> LoopRegister = gematria::getUnusedGPRegister(
      *DisInstructions, State.getRegInfo(), State.getInstrInfo());

  if (LoopRegister.has_value()) {
    MemAnnotations.set_loop_register(State.getRegInfo().getName(*LoopRegister));
  }

  MemAnnotations.set_block_contents(kInitialMemVal);

  return MemAnnotations;
}

}  // namespace gematria
