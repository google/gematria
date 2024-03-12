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

#include <map>

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86RegisterInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"

using namespace llvm;

namespace gematria {

unsigned getSuperRegister(unsigned OriginalRegister,
                          const MCRegisterInfo &RegisterInfo) {
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

std::vector<unsigned> BasicBlockUtils::getUsedRegisters(
    const std::vector<DisassembledInstruction> &Instructions,
    const MCRegisterInfo &RegisterInfo, const MCInstrInfo &InstructionInfo) {
  std::map<unsigned, bool> UsedInputRegisters;
  std::map<unsigned, bool> DefinedInBBRegisters;
  for (const gematria::DisassembledInstruction &Instruction : Instructions) {
    unsigned InstructionDefs =
        InstructionInfo.get(Instruction.mc_inst.getOpcode()).getNumDefs();
    for (unsigned OperandIndex = InstructionDefs;
         OperandIndex < Instruction.mc_inst.getNumOperands(); ++OperandIndex) {
      if (Instruction.mc_inst.getOperand(OperandIndex).isReg()) {
        unsigned RegisterNumber =
            Instruction.mc_inst.getOperand(OperandIndex).getReg();
        if (RegisterNumber == 0) continue;
        unsigned SuperRegister = getSuperRegister(RegisterNumber, RegisterInfo);
        // If the register was already defined within the basic block, we don't
        // need to define it ourselves.
        if (DefinedInBBRegisters.count(SuperRegister) > 0) continue;
        UsedInputRegisters[SuperRegister] = true;
      }
    }
    // We also need to handle instructions that have implict uses.
    for (unsigned ImplicitlyUsedRegister :
         InstructionInfo.get(Instruction.mc_inst.getOpcode()).implicit_uses()) {
      UsedInputRegisters[getSuperRegister(ImplicitlyUsedRegister,
                                          RegisterInfo)] = true;
    }
    for (unsigned OperandIndex = 0; OperandIndex < InstructionDefs;
         ++OperandIndex) {
      if (Instruction.mc_inst.getOperand(OperandIndex).isReg()) {
        unsigned RegisterNumber =
            Instruction.mc_inst.getOperand(OperandIndex).getReg();
        if (RegisterNumber == 0) continue;
        DefinedInBBRegisters[getSuperRegister(RegisterNumber, RegisterInfo)] =
            true;
      }
    }
  }

  std::vector<unsigned> UsedRegistersList;
  UsedRegistersList.reserve(UsedInputRegisters.size());

  for (const auto [RegisterIndex, RegisterUsed] : UsedInputRegisters)
    UsedRegistersList.push_back(RegisterIndex);

  return UsedRegistersList;
}

unsigned getSuperRegisterAllClasses(unsigned OriginalRegister,
                                    const MCRegisterInfo &RegisterInfo) {
  unsigned SuperRegister = OriginalRegister;
  for (MCPhysReg CurrentSuperRegister :
       RegisterInfo.superregs_inclusive(OriginalRegister))
    SuperRegister = CurrentSuperRegister;
  return SuperRegister;
}

std::optional<unsigned> BasicBlockUtils::getLoopRegister(
    const std::vector<DisassembledInstruction> &Instructions,
    const MCRegisterInfo &RegisterInfo, const MCInstrInfo &InstructionInfo) {
  std::map<unsigned, bool> UsedRegisters;
  for (const gematria::DisassembledInstruction &Instruction : Instructions) {
    for (unsigned OperandIndex = 0;
         OperandIndex < Instruction.mc_inst.getNumOperands(); ++OperandIndex) {
      if (Instruction.mc_inst.getOperand(OperandIndex).isReg()) {
        unsigned RegisterNumber =
            Instruction.mc_inst.getOperand(OperandIndex).getReg();
        if (RegisterNumber == 0) continue;
        UsedRegisters[getSuperRegisterAllClasses(RegisterNumber,
                                                 RegisterInfo)] = true;
      }
    }
    for (unsigned ImplicitlyUsedRegister :
         InstructionInfo.get(Instruction.mc_inst.getOpcode()).implicit_uses())
      UsedRegisters[getSuperRegisterAllClasses(ImplicitlyUsedRegister,
                                               RegisterInfo)] = true;
    for (unsigned ImplicitlyDefinedRegister :
         InstructionInfo.get(Instruction.mc_inst.getOpcode()).implicit_defs())
      UsedRegisters[getSuperRegisterAllClasses(ImplicitlyDefinedRegister,
                                               RegisterInfo)] = true;
  }

  const auto &GR64RegisterClass =
      RegisterInfo.getRegClass(X86::GR64_NOREX2RegClassID);
  std::optional<unsigned> LoopRegister = std::nullopt;
  for (unsigned I = 0; I < GR64RegisterClass.getNumRegs(); ++I) {
    unsigned CurrentRegister = GR64RegisterClass.getRegister(I);
    if (CurrentRegister == X86::RIP) continue;
    if (UsedRegisters.count(CurrentRegister) == 0) {
      LoopRegister = CurrentRegister;
      break;
    }
  }

  return LoopRegister;
}

}  // namespace gematria
