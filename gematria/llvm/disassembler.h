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

#ifndef THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_DISASSEMBLER_H_
#define THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_DISASSEMBLER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Error.h"

namespace gematria {

// The result of disassembling an instruction. Contains the data of the
// instruction in the form of the output llvm::MCInst and the information needed
// to fill in a MachineInstructionProto.
struct DisassembledInstruction {
  // The address of the instruction.
  uint64_t address = 0;
  // The assembly code of the instruction.
  std::string assembly;
  // The binary code of the instruction.
  std::string machine_code;

  llvm::MCInst mc_inst;
};

// Creates the assembly representation of an llvm::MCInst.
std::string AssemblyFromMCInst(const llvm::MCInstrInfo& instruction_info,
                               const llvm::MCRegisterInfo& register_info,
                               const llvm::MCSubtargetInfo& subtarget_info,
                               llvm::MCInstPrinter& printer,
                               const llvm::MCInst& instruction);

// Disassembles at most one instruction starting at the first byte of
// `machine_code`. On success, returns the data of the instruction, and removes
// all its bytes from `machine_code`. Uses base_address as the address of the
// instruction in the output proto.
// On error, returns an error message and leaves `machine_code` unchanged.
llvm::Expected<DisassembledInstruction> DisassembleOneInstruction(
    const llvm::MCDisassembler& disassembler,
    const llvm::MCInstrInfo& instruction_info,
    const llvm::MCRegisterInfo& register_info,
    const llvm::MCSubtargetInfo& subtarget_info, llvm::MCInstPrinter& printer,
    uint64_t base_address, llvm::ArrayRef<uint8_t>& machine_code);

// Disassembles all instructions from `machine_code`. Succeeds only when all
// bytes have been consumed, i.e. it is an error if some number of bytes towards
// the end can't be disassembled. Uses base_address as the address of the first
// instruction in the block. The address of the following instruction is
// computed as the sum of the base address and the size of the instructions that
// precede it in the block.
llvm::Expected<std::vector<DisassembledInstruction>> DisassembleAllInstructions(
    const llvm::MCDisassembler& disassembler,
    const llvm::MCInstrInfo& instruction_info,
    const llvm::MCRegisterInfo& register_info,
    const llvm::MCSubtargetInfo& subtarget_info, llvm::MCInstPrinter& printer,
    uint64_t base_address, llvm::ArrayRef<uint8_t> machine_code);

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_DISASSEMBLER_H_
