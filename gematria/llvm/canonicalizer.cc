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

#include <cassert>
#include <string>
#include <string_view>

#include "gematria/basic_block/basic_block.h"
#include "lib/Target/X86/MCTargetDesc/X86BaseInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include <sstream>


#ifdef DEBUG
#define LOG(X) \
  llvm::errs() << X << "\n"
#else
#define LOG(X)
#endif

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

void ReplaceExprOperands(llvm::MachineInstr& instruction) {
  for (int i = 0; i < instruction.getNumOperands(); ++i) {
    llvm::MachineOperand& operand = instruction.getOperand(i);
    if (operand.isSymbol() || operand.isGlobal() || operand.isCPI()) {
      operand = llvm::MachineOperand::CreateImm(1);
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

Instruction Canonicalizer::InstructionFromMachineInstr(llvm::MachineInstr& MI) const {
  ReplaceExprOperands(MI);
  return PlatformSpecificInstructionFromMachineInstr(MI);
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

// return true if it's pyhsical register
bool Canonicalizer::GetRegisterNameOrEmpty(
    const llvm::MachineOperand& operand, std::string& name, size_t& size) const {
  assert(operand.isReg());
  // cast operand to llvm::Register
  const llvm::Register& reg = operand.getReg();
  if (reg.isVirtual()){
    const llvm::MachineFunction *MF = operand.getParent()->getParent()->getParent();
    const llvm::TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
    const llvm::MachineRegisterInfo &MRI = MF->getRegInfo();
    size_t Size = TRI->getRegSizeInBits(reg, MRI);
    name = "%" + std::to_string(llvm::Register::virtReg2Index(reg));
    size = Size;
    return false;
  } else {
    name = target_machine_.getMCRegisterInfo()->getName(reg);
    return true;
  }
}

namespace {

llvm::SmallVector<std::string_view, 2> SplitByAny(std::string_view str,
                                                  std::string_view separators) {
  llvm::SmallVector<std::string_view, 2> tokens;
  size_t pos = str.find_first_not_of(separators);

  while (pos != str.npos) {
    const size_t next_separator = str.find_first_of(separators, pos);
    size_t token_end = std::min(next_separator, str.size());
    tokens.push_back(str.substr(pos, token_end));

    pos = str.find_first_not_of(separators, next_separator);
  }

  return tokens;
}

void AddX86VendorMnemonicAndPrefixes(
    llvm::MCInstPrinter& printer, const llvm::MCSubtargetInfo& subtarget_info,
    const llvm::MCInst& mcinst, Instruction& instruction) {
  constexpr const char* kKnownPrefixes[] = {"nofpexcept"};

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

void AddMIRVendorMnemonicAndPrefixes(
    const llvm::MCSubtargetInfo& subtarget_info,
    const llvm::MachineInstr& MI, Instruction& instruction) {
  constexpr const char* kKnownPrefixes[] = {"REP", "LOCK", "REPNE", "REPE"};

  std::string assembly_code;
  llvm::raw_string_ostream stream(assembly_code);
  MI.print(stream);
  stream.flush();

  // TODO: now used a dummy way to seperate tokens... don't know why SplitByAny does 
  // not work for MIR stream
  llvm::SmallVector<std::string, 2> tokens;
  tokens.push_back("");
  for (char c : assembly_code){
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r'){
      tokens.push_back("");
    } else {
      tokens.back().push_back(c);
    }
  }
  assert(!tokens.empty());

  // If there is only one token, we treat it as the mnemonic no matter what.
  if (tokens.size() == 1) {
    instruction.mnemonic = tokens[0];
    return;
  }

  // Otherwise, we strip known prefixes and treat the first token that is not a
  // prefix as the mnemonic.
  // If MI has result register, then the first token prefix/mnemonic is the last result register index + 2
  size_t result_reg_index = 0;
  bool is_first_time = true;
  for (uint i = 0; i < MI.getNumOperands(); i++){
    auto& MOP = MI.getOperand(i);
    if (MOP.isReg() && !MOP.isImplicit() && MOP.isDef()){
      if (is_first_time){
        is_first_time = false;
        result_reg_index = result_reg_index + 2;
      } else {
        result_reg_index = result_reg_index + 1;
      }
    }
  }

  for (const llvm::StringRef token : tokens) {
    if (result_reg_index > 0){
      result_reg_index = result_reg_index - 1;
      continue;
    }

    // process prefix and mnemonic tokens
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


Instruction X86Canonicalizer::PlatformSpecificInstructionFromMachineInstr(const llvm::MachineInstr & MI) const {
  // NOTE (lukezhuz): Memory alias are determined by the type of memory operand
  // if it's a FrameIndex, then different frameindex should not be aliased to each other
  
  // use -1 denote global/other memory alias; use other to denote stack memory alias
  constexpr int kWholeMemoryAliasGroup = 1;

  const llvm::MCRegisterInfo& register_info =
      *target_machine_.getMCRegisterInfo();
  const llvm::MCInstrInfo& instr_info = *target_machine_.getMCInstrInfo();

  Instruction instruction;
  instruction.llvm_mnemonic =
      target_machine_.getMCInstrInfo()->getName(MI.getOpcode());
  instruction.mnemonic =
      target_machine_.getMCInstrInfo()->getName(MI.getOpcode());

  const llvm::MCInstrDesc& descriptor = instr_info.get(MI.getOpcode());
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
    AddOperand(MI, /*operand_index=*/operand_index,
               /*is_output_operand=*/is_output_operand,
               /*is_address_computation_tuple=*/is_address_computation_tuple,
               instruction,
               descriptor);
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

void X86Canonicalizer::AddOperand(const llvm::MachineInstr& mi, int operand_index,
                                  bool is_output_operand,
                                  bool is_address_computation_tuple,
                                  Instruction& instruction,
                                  const llvm::MCInstrDesc& descriptor) const {
  assert(operand_index < mi.getNumOperands());
  assert(!is_address_computation_tuple ||
         (operand_index + 5 <= mi.getNumOperands()));

  const llvm::MachineOperand& operand = mi.getOperand(operand_index);
  // Skip empty register operand, but not if they are part of a memory 5-tuple.
  // Empty register in a memory 5-tuple is for when the address computation uses
  // only a subset of components.
  if (!is_address_computation_tuple && operand.isReg() && operand.getReg() == 0)
    return;
  
  std::vector<InstructionOperand>& operand_list =
      is_output_operand ? instruction.output_operands
                        : instruction.input_operands;
  if (is_address_computation_tuple) { // TODO: Check if MIR has address computation tuple
    std::string base_register;
    size_t base_register_size = 64;
    size_t index_register_size = 64;
    size_t segment_register_size = 64;
    if (mi.getOperand(operand_index + llvm::X86::AddrBaseReg).isReg()){
      GetRegisterNameOrEmpty(
        mi.getOperand(operand_index + llvm::X86::AddrBaseReg), base_register, base_register_size);
    } else if (mi.getOperand(operand_index + llvm::X86::AddrBaseReg).isFI()){
      base_register = "RBP";
    } else {
      assert(false && "unsupported base register type");
      LOG(mi);
    }
      const int64_t displacement =
          mi.getOperand(operand_index + llvm::X86::AddrDisp).getImm();
      std::string index_register;
      GetRegisterNameOrEmpty(
        mi.getOperand(operand_index + llvm::X86::AddrIndexReg), index_register, index_register_size);
      const int64_t scaling =
          mi.getOperand(operand_index + llvm::X86::AddrScaleAmt).getImm();
      std::string segment_register; 
      GetRegisterNameOrEmpty(
          mi.getOperand(operand_index + llvm::X86::AddrSegmentReg), segment_register, segment_register_size);
      operand_list.push_back(InstructionOperand::Address(
          /* base_register= */ std::move(base_register),
          /* displacement= */ displacement,
          /* index_register= */ std::move(index_register),
          /* scaling= */ static_cast<int>(scaling),
          /* segment_register= */ std::move(segment_register),
          /* base_register_size= */ base_register_size,
          /* index_register_size= */ index_register_size,
          /* segment_register_size= */ segment_register_size));
        LOG("Hit here address_computation_tuple reg " << mi << "\n");
  } else if (operand.isReg()) {
    std::string name;
    size_t size;
    bool is_physical_reg = GetRegisterNameOrEmpty(operand, name, size);
    if (is_physical_reg){
      operand_list.push_back(
          InstructionOperand::Register(name));
    } else {
      operand_list.push_back(
          InstructionOperand::VirtualRegister(name, size, {}));
    }
  } else if (operand.isImm()) {
    operand_list.push_back(
        InstructionOperand::ImmediateValue(operand.getImm()));
  } else if (operand.isCImm()) {
    operand_list.push_back(
        InstructionOperand::ImmediateValue(operand.getCImm()->getZExtValue()));
  } else if (operand.isFPImm()) {
    operand_list.push_back(InstructionOperand::FpImmediateValue(
        llvm::bit_cast<double>(operand.getFPImm())));
  } else {
    llvm::errs() << "Unsupported operand type: ";
    operand.print(llvm::errs());
    llvm::errs() << "\n";
    instruction.is_valid = false;
  }
}

}  // namespace gematria
