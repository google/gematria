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

#include "gematria/datasets/basic_block_utils.h"

#include <set>

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86RegisterInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"

using namespace llvm;

namespace gematria {

unsigned getSuperRegister(unsigned OriginalRegister,
                          const MCRegisterInfo &RegisterInfo) {
  // We should be able to get the super register by getting an iterator from
  // RegisterInfo and getting the back, but this always returns a value of
  // zero.
  // TODO(boomanaiden154): Investigate why this is ocurring on the LLVM side
  // and fix it.
  unsigned SuperRegister = OriginalRegister;
  for (MCPhysReg CurrentSuperRegister :
       RegisterInfo.superregs_inclusive(OriginalRegister)) {
    SuperRegister = CurrentSuperRegister;
  }
  // Only return super registers for GPRs. Since mainly GPRs will be used for
  // addressing, redefining other aliasing registers (like vector registers)
  // does not matter as much.
  if (RegisterInfo.getRegClass(X86::GR64RegClassID).contains(SuperRegister))
    return SuperRegister;
  else
    return OriginalRegister;
}

std::vector<unsigned> getUsedRegisters(
    const ArrayRef<DisassembledInstruction> Instructions,
    const MCRegisterInfo &RegisterInfo, const MCInstrInfo &InstructionInfo) {
  std::set<unsigned> UsedInputRegisters;
  std::set<unsigned> DefinedInBBRegisters;
  for (const gematria::DisassembledInstruction &Instruction : Instructions) {
    unsigned InstructionDefs =
        InstructionInfo.get(Instruction.mc_inst.getOpcode()).getNumDefs();
    for (unsigned OperandIndex = InstructionDefs;
         OperandIndex < Instruction.mc_inst.getNumOperands(); ++OperandIndex) {
      const MCOperand &CurrentOperand =
          Instruction.mc_inst.getOperand(OperandIndex);
      if (CurrentOperand.isReg()) {
        unsigned RegisterNumber = CurrentOperand.getReg();
        if (RegisterNumber == 0) continue;
        unsigned SuperRegister = getSuperRegister(RegisterNumber, RegisterInfo);
        // If the register was already defined within the basic block, we don't
        // need to define it ourselves.
        if (DefinedInBBRegisters.count(SuperRegister) > 0) continue;
        UsedInputRegisters.insert(SuperRegister);
      }
    }
    // We also need to handle instructions that have implict uses.
    for (unsigned ImplicitlyUsedRegister :
         InstructionInfo.get(Instruction.mc_inst.getOpcode()).implicit_uses()) {
      UsedInputRegisters.insert(
          getSuperRegister(ImplicitlyUsedRegister, RegisterInfo));
    }
    // Handle instructions that have implicit defs
    for (unsigned ImplicitlyDefinedRegister :
         InstructionInfo.get(Instruction.mc_inst.getOpcode()).implicit_defs()) {
      DefinedInBBRegisters.insert(
          getSuperRegister(ImplicitlyDefinedRegister, RegisterInfo));
    }
    for (unsigned OperandIndex = 0; OperandIndex < InstructionDefs;
         ++OperandIndex) {
      if (Instruction.mc_inst.getOperand(OperandIndex).isReg()) {
        unsigned RegisterNumber =
            Instruction.mc_inst.getOperand(OperandIndex).getReg();
        if (RegisterNumber == 0) continue;
        DefinedInBBRegisters.insert(
            getSuperRegister(RegisterNumber, RegisterInfo));
      }
    }
  }

  std::vector<unsigned> UsedRegistersList;
  UsedRegistersList.reserve(UsedInputRegisters.size());

  UsedRegistersList.assign(UsedInputRegisters.begin(),
                           UsedInputRegisters.end());

  return UsedRegistersList;
}

}  // namespace gematria
