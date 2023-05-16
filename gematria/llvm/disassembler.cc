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

#include "gematria/llvm/disassembler.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"
#include "llvm/include/llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/include/llvm/MC/MCInstPrinter.h"
#include "llvm/include/llvm/MC/MCInstrInfo.h"
#include "llvm/include/llvm/MC/MCRegisterInfo.h"
#include "llvm/include/llvm/MC/MCSubtargetInfo.h"
#include "llvm/include/llvm/Support/raw_ostream.h"

namespace gematria {

std::string AssemblyFromMCInst(const llvm::MCInstrInfo& instruction_info,
                               const llvm::MCRegisterInfo& register_info,
                               const llvm::MCSubtargetInfo& subtarget_info,
                               llvm::MCInstPrinter& printer,
                               const llvm::MCInst& instruction) {
  std::string assembly_code;
  llvm::raw_string_ostream stream(assembly_code);
  printer.printInst(&instruction, 0, "", subtarget_info, stream);
  stream.flush();
  return assembly_code;
}

absl::StatusOr<DisassembledInstruction> DisassembleOneInstruction(
    const llvm::MCDisassembler& disassembler,
    const llvm::MCInstrInfo& instruction_info,
    const llvm::MCRegisterInfo& register_info,
    const llvm::MCSubtargetInfo& subtarget_info, llvm::MCInstPrinter& printer,
    uint64_t base_address, absl::Span<const uint8_t>& machine_code) {
  if (machine_code.empty())
    return absl::InvalidArgumentError("The input is empty");

  std::string disassembler_output_buffer;
  llvm::raw_string_ostream output(disassembler_output_buffer);

  DisassembledInstruction result;
  llvm::ArrayRef<uint8_t> data(machine_code.data(), machine_code.size());
  // Use the current position of the instruction in memory as its address. This
  // is most likely not the "true" address, but in most cases it's the best we
  // get, and it is the address at which the instruction is parsed.
  const uint64_t instruction_address =
      reinterpret_cast<uint64_t>(machine_code.data());
  uint64_t instruction_size = 0;
  result.instruction.set_address(base_address);
  using DecodeStatus = llvm::MCDisassembler::DecodeStatus;
  const DecodeStatus status = disassembler.getInstruction(
      result.mc_inst, instruction_size, data, instruction_address, output);
  switch (status) {
    case DecodeStatus::Success:
      break;
    case DecodeStatus::Fail:
      return absl::InvalidArgumentError(
          absl::StrCat("Disassembling the instruction failed: ",
                       disassembler_output_buffer));
    case DecodeStatus::SoftFail:
      return absl::InvalidArgumentError(
          absl::StrCat("Incomplete instruction: ", disassembler_output_buffer));
  }

  if (instruction_size > machine_code.size()) {
    return absl::InternalError(absl::StrCat(
        "The instruction size (", instruction_size,
        ") is bigger than the input buffer (", machine_code.size(), ")."));
  }

  std::string instruction_machine_code(machine_code.begin(),
                                       machine_code.begin() + instruction_size);
  result.instruction.set_machine_code(std::move(instruction_machine_code));
  result.instruction.set_assembly(
      AssemblyFromMCInst(instruction_info, register_info, subtarget_info,
                         printer, result.mc_inst));
  machine_code.remove_prefix(instruction_size);
  return result;
}

absl::StatusOr<std::vector<DisassembledInstruction>> DisassembleAllInstructions(
    const llvm::MCDisassembler& disassembler,
    const llvm::MCInstrInfo& instruction_info,
    const llvm::MCRegisterInfo& register_info,
    const llvm::MCSubtargetInfo& subtarget_info, llvm::MCInstPrinter& printer,
    uint64_t base_address, absl::Span<const uint8_t> machine_code) {
  std::vector<DisassembledInstruction> result;

  int num_consumed_bytes = 0;
  while (!machine_code.empty()) {
    absl::StatusOr<DisassembledInstruction> instruction_or_status =
        DisassembleOneInstruction(
            disassembler, instruction_info, register_info, subtarget_info,
            printer, base_address + num_consumed_bytes, machine_code);
    if (!instruction_or_status.ok()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Parsing of machine code failed at byte ", num_consumed_bytes,
          " with error: ", instruction_or_status.status().ToString()));
    }
    result.push_back(std::move(instruction_or_status).value());
    num_consumed_bytes += result.back().instruction.machine_code().size();
  }

  return std::move(result);
}

}  // namespace gematria
