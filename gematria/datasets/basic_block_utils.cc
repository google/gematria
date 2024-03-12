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

#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"

using namespace llvm;

namespace gematria {

std::vector<unsigned> BasicBlockUtils::getUsedRegisters(
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
        unsigned SuperRegisterNumber = 0;
        for (MCPhysReg CurrentSuperRegister :
             RegisterInfo.superregs_inclusive(RegisterNumber)) {
          SuperRegisterNumber = CurrentSuperRegister;
        }
        UsedRegisters[SuperRegisterNumber] = true;
      }
    }
  }

  std::vector<unsigned> UsedRegistersList;
  UsedRegistersList.reserve(UsedRegisters.size());

  for (const auto [RegisterIndex, RegisterUsed] : UsedRegisters)
    UsedRegistersList.push_back(RegisterIndex);

  return UsedRegistersList;
}

}  // namespace gematria
