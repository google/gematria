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

#include "gematria/llvm/canonicalizer.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gematria/basic_block/basic_block.h"
#include "lib/Target/X86/MCTargetDesc/X86BaseInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/bit.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

namespace gematria {
namespace {

// Replace `expr` operands in the instructions. The canonicalization of operands
// and the instruction printer used to get the prefix of the instruction can't
// handle them without additional information.
//
// Expr operands are immediate values that are computed based on the layout of
// the binary code, e.g. addresses relative to labels or symbols. We can't
// evaluate them without laying out the binary code, so we just replace them
// with a constant.
void ReplaceExprOperands(llvm::MCInst& instruction) {
  for (int i = 0; i < instruction.getNumOperands(); ++i) {
    llvm::MCOperand& operand = instruction.getOperand(i);
    if (operand.isExpr()) {
      // TODO(ondrasej): In some cases the value may change the binary encoding
      // of the instruction, e.g. switch between an 8-bit or a 32-bit encoding
      // of the displacement and 0 might have a special meaning (e.g. do not use
      // displacement at all).
      operand = llvm::MCOperand::createImm(1);
    }
  }
}

}  // namespace

Canonicalizer::Canonicalizer(const llvm::TargetMachine* target_machine)
    : target_machine_(*target_machine) {
  assert(target_machine != nullptr);
}

Canonicalizer::~Canonicalizer() = default;

Instruction Canonicalizer::InstructionFromMCInst(llvm::MCInst mcinst) const {
  ReplaceExprOperands(mcinst);
  return PlatformSpecificInstructionFromMCInst(mcinst);
}

BasicBlock Canonicalizer::BasicBlockFromMCInst(
    llvm::ArrayRef<llvm::MCInst> mcinsts) const {
  BasicBlock block;
  for (const llvm::MCInst& mcinst : mcinsts) {
    block.instructions.push_back(InstructionFromMCInst(mcinst));
  }
  return block;
}

std::string Canonicalizer::GetRegisterNameOrEmpty(
    const llvm::MCOperand& operand) const {
  assert(operand.isReg());
  return target_machine_.getMCRegisterInfo()->getName(operand.getReg());
}

namespace {

llvm::SmallVector<std::string_view, 2> SplitByAny(std::string_view str,
                                                  std::string_view separators) {
  llvm::SmallVector<std::string_view, 2> tokens;
  size_t pos = str.find_first_not_of(separators);

  while (pos != str.npos) {
    const size_t next_separator = str.find_first_of(separators, pos);
    tokens.push_back(str.substr(pos, next_separator - pos));

    pos = str.find_first_not_of(separators, next_separator);
  }

  return tokens;
}

void AddX86VendorMnemonicAndPrefixes(
    llvm::MCInstPrinter& printer, const llvm::MCSubtargetInfo& subtarget_info,
    const llvm::MCInst& mcinst, Instruction& instruction) {
  constexpr const char* kKnownPrefixes[] = {"REP", "LOCK", "REPNE", "REPE"};

  std::string assembly_code;
  llvm::raw_string_ostream stream(assembly_code);
  printer.printInst(&mcinst, 0, "", subtarget_info, stream);
  stream.flush();

  auto tokens = SplitByAny(assembly_code, "\t\r\n ");
  assert(!tokens.empty());

  // If there is only one token, we treat it as the mnemonic no matter what.
  if (tokens.size() == 1) {
    instruction.mnemonic = tokens[0];
    return;
  }

  // Otherwise, we strip known prefixes and treat the first token that is not a
  // prefix as the mnemonic.
  for (const llvm::StringRef token : tokens) {
    std::string uppercased_token = token.trim().upper();
    const bool is_known_prefix =
        std::find(std::begin(kKnownPrefixes), std::end(kKnownPrefixes),
                  uppercased_token) != std::end(kKnownPrefixes);
    if (is_known_prefix) {
      instruction.prefixes.push_back(std::move(uppercased_token));
    } else {
      instruction.mnemonic = std::move(uppercased_token);
      break;
    }
  }
  assert(!instruction.mnemonic.empty());
}

int GetX86MemoryOperandPosition(const llvm::MCInstrDesc& descriptor) {
  const int memory_operand_no =
      llvm::X86II::getMemoryOperandNo(descriptor.TSFlags);
  // Return early if the instruction does not use the memory 5-tuple.
  if (memory_operand_no == -1) return -1;
  return memory_operand_no +
         static_cast<int>(llvm::X86II::getOperandBias(descriptor));
}

}  // namespace

X86Canonicalizer::X86Canonicalizer(const llvm::TargetMachine* target_machine)
    : Canonicalizer(target_machine) {
  static constexpr int kIntelSyntax = 1;
  const llvm::Target& target = target_machine->getTarget();
  mcinst_printer_.reset(target.createMCInstPrinter(
      target_machine_.getTargetTriple(), kIntelSyntax,
      *target_machine_.getMCAsmInfo(), *target_machine_.getMCInstrInfo(),
      *target_machine_.getMCRegisterInfo()));
}

X86Canonicalizer::~X86Canonicalizer() = default;

Instruction X86Canonicalizer::PlatformSpecificInstructionFromMCInst(
    const llvm::MCInst& mcinst) const {
  // NOTE(ondrasej): For now, we assume that all memory references are aliased.
  // This is an overly conservative but safe choice. Note that Ithemal chose the
  // other extreme where no two memory accesses are aliased - we may want to
  // support this use case too.
  constexpr int kWholeMemoryAliasGroup = 1;

  const llvm::MCRegisterInfo& register_info =
      *target_machine_.getMCRegisterInfo();
  const llvm::MCInstrInfo& instr_info = *target_machine_.getMCInstrInfo();

  Instruction instruction;
  instruction.llvm_mnemonic =
      target_machine_.getMCInstrInfo()->getName(mcinst.getOpcode());
  AddX86VendorMnemonicAndPrefixes(*mcinst_printer_,
                                  *target_machine_.getMCSubtargetInfo(), mcinst,
                                  instruction);

  const llvm::MCInstrDesc& descriptor = instr_info.get(mcinst.getOpcode());
  if (descriptor.mayLoad()) {
    instruction.input_operands.push_back(
        InstructionOperand::MemoryLocation(kWholeMemoryAliasGroup));
  }
  if (descriptor.mayStore()) {
    instruction.output_operands.push_back(
        InstructionOperand::MemoryLocation(kWholeMemoryAliasGroup));
  }

  const int memory_operand_index = GetX86MemoryOperandPosition(descriptor);
  for (int operand_index = 0; operand_index < descriptor.getNumOperands();
       ++operand_index) {
    const bool is_output_operand = operand_index < descriptor.getNumDefs();
    const bool is_address_computation_tuple =
        operand_index == memory_operand_index;
    AddOperand(mcinst, /*operand_index=*/operand_index,
               /*is_output_operand=*/is_output_operand,
               /*is_address_computation_tuple=*/is_address_computation_tuple,
               instruction);
    if (is_address_computation_tuple) {
      // A memory reference is represented as a 5-tuple. The whole 5-tuple is
      // processed in one CanonicalizeOperand() call and we need to skip the
      // remaining 4 elements here.
      operand_index += 4;
    }
  }

  for (llvm::MCPhysReg implicit_output_register : descriptor.implicit_defs()) {
    instruction.implicit_output_operands.push_back(InstructionOperand::Register(
        register_info.getName(implicit_output_register)));
  }
  for (llvm::MCPhysReg implicit_input_register : descriptor.implicit_uses()) {
    instruction.implicit_input_operands.push_back(InstructionOperand::Register(
        register_info.getName(implicit_input_register)));
  }

  return instruction;
}

void X86Canonicalizer::AddOperand(const llvm::MCInst& mcinst, int operand_index,
                                  bool is_output_operand,
                                  bool is_address_computation_tuple,
                                  Instruction& instruction) const {
  assert(operand_index < mcinst.getNumOperands());
  assert(!is_address_computation_tuple ||
         (operand_index + 5 <= mcinst.getNumOperands()));

  const llvm::MCOperand& operand = mcinst.getOperand(operand_index);
  // Skip empty register operand, but not if they are part of a memory 5-tuple.
  // Empty register in a memory 5-tuple is for when the address computation uses
  // only a subset of components.
  if (!is_address_computation_tuple && operand.isReg() && operand.getReg() == 0)
    return;

  std::vector<InstructionOperand>& operand_list =
      is_output_operand ? instruction.output_operands
                        : instruction.input_operands;
  if (is_address_computation_tuple) {
    std::string base_register = GetRegisterNameOrEmpty(
        mcinst.getOperand(operand_index + llvm::X86::AddrBaseReg));
    const int64_t displacement =
        mcinst.getOperand(operand_index + llvm::X86::AddrDisp).getImm();
    std::string index_register = GetRegisterNameOrEmpty(
        mcinst.getOperand(operand_index + llvm::X86::AddrIndexReg));
    const int64_t scaling =
        mcinst.getOperand(operand_index + llvm::X86::AddrScaleAmt).getImm();
    std::string segment_register = GetRegisterNameOrEmpty(
        mcinst.getOperand(operand_index + llvm::X86::AddrSegmentReg));
    operand_list.push_back(InstructionOperand::Address(
        /* base_register= */ std::move(base_register),
        /* displacement= */ displacement,
        /* index_register= */ std::move(index_register),
        /* scaling= */ static_cast<int>(scaling),
        /* segment_register= */ std::move(segment_register)));
  } else if (operand.isReg()) {
    operand_list.push_back(
        InstructionOperand::Register(GetRegisterNameOrEmpty(operand)));
  } else if (operand.isImm()) {
    operand_list.push_back(
        InstructionOperand::ImmediateValue(operand.getImm()));
  } else if (operand.isDFPImm()) {
    operand_list.push_back(InstructionOperand::FpImmediateValue(
        llvm::bit_cast<double>(operand.getDFPImm())));
  } else if (operand.isSFPImm()) {
    operand_list.push_back(
        InstructionOperand::FpImmediateValue(operand.getSFPImm()));
  } else {
    llvm::errs() << "Unsupported operand type: ";
    operand.print(llvm::errs());
    llvm::errs() << "\n";
    assert(false);
  }
}

}  // namespace gematria
