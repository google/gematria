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

// Functions for extracting basic blocks from LLVM data structures. These
// functions are intentionally kept in a separate library to isolate the LLVM
// dependences to the smallest scope possible.

#ifndef THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_CANONICALIZER_H_
#define THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_CANONICALIZER_H_

#include <memory>
#include <string>

#include "gematria/basic_block/basic_block.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/MachineBasicBlock.h"

namespace gematria {

// Abstract interface for code that extracts basic block data structures from
// binary machine code. Each supported platform should provide its own subclass
// that implements extraction for this specific platform.
class Canonicalizer {
 public:
  explicit Canonicalizer(const llvm::TargetMachine* target_machine);
  virtual ~Canonicalizer();

  // Extracts data from a single machine instruction.
  virtual Instruction InstructionFromMCInst(llvm::MCInst mcinst) const;

  // Extract data from a single MachineInstr (MIR)
  virtual Instruction InstructionFromMachineInstr(
        const llvm::MachineInstr& machine_instr) const;

  // Extracts data from a sequence of instructions.
  virtual BasicBlock BasicBlockFromMCInst(
      llvm::ArrayRef<llvm::MCInst> mcinsts) const;

  // Returns the target machine on which the canonicalizer is based.
  const llvm::TargetMachine& target_machine() const { return target_machine_; }

 protected:
  // The platform-specific code for instruction extraction. When called, this
  // method can assume that `instruction` does not have any expression operands.
  virtual Instruction PlatformSpecificInstructionFromMCInst(
      const llvm::MCInst& instruction) const = 0;

  // The platform-specific code for instruction extraction at MIR level. When called, this
  // method can assume that `instruction` does not have any expression operands.
  virtual Instruction PlatformSpecificInstructionFromMachineInstr(
      const llvm::MachineInstr& instruction) const = 0;

  // Returns the name of a register in an operand. Returns an empty string when
  // the operand is an "undefined" operand.
  // This method must not be called when `operand.isReg()` is false.
  std::string GetRegisterNameOrEmpty(const llvm::MCOperand& operand) const;

  const llvm::TargetMachine& target_machine_;
};

// A version of basic block extractor for X86-64.
class X86Canonicalizer final : public Canonicalizer {
 public:
  explicit X86Canonicalizer(const llvm::TargetMachine* target_machine);
  ~X86Canonicalizer() override;

 private:
  Instruction PlatformSpecificInstructionFromMCInst(
      const llvm::MCInst& mcinst) const override;
  Instruction PlatformSpecificInstructionFromMachineInstr(
      const llvm::MachineInstr& MI) const override;

  void AddOperand(const llvm::MCInst& mcinst, int operand_index,
                  bool is_output_operand, bool is_address_computation_tuple,
                  Instruction& instruction) const;

  std::unique_ptr<llvm::MCInstPrinter> mcinst_printer_;
};

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_CANONICALIZER_H_
