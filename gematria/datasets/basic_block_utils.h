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

// Contains utilities for analying and manipulating basic blocks.

#ifndef THIRD_PARTY_GEMATRIA_GEMATRIA_DATASETS_BASIC_BLOCK_UTILS_H_
#define THIRD_PARTY_GEMATRIA_GEMATRIA_DATASETS_BASIC_BLOCK_UTILS_H_

#include <vector>

#include "gematria/llvm/disassembler.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"

namespace gematria {

using namespace llvm;

// Gets the registers used by a basic block (in the form of a sequence of
// instructions).
std::vector<unsigned> getUsedRegisters(
    const ArrayRef<DisassembledInstruction> Instructions,
    const MCRegisterInfo &RegisterInfo, const MCInstrInfo &InstructionInfo);

// Returns a GPR not touched by the block. This is intended to be used to find
// a register that can be used to store a loop iteration index. In the case
// there is no free GPR, the function will returns std::nullopt.
std::optional<unsigned> getUnusedGPRegister(
    const std::vector<DisassembledInstruction> &Instructions,
    const MCRegisterInfo &RegisterInfo, const MCInstrInfo &InstructionInfo);

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_DATASETS_BASIC_BLOCK_UTILS_H_
