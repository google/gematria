// Copyright 2022 Google Inc.
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

#include "gematria/basic_block/basic_block.h"

#include <cassert>
#include <ios>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace gematria {

#define GEMATRIA_PRINT_ENUM_VALUE_TO_OS(os, enum_value) \
  case enum_value:                                      \
    os << #enum_value;                                  \
    break

std::ostream& operator<<(std::ostream& os, OperandType operand_type) {
  switch (operand_type) {
    GEMATRIA_PRINT_ENUM_VALUE_TO_OS(os, OperandType::kUnknown);
    GEMATRIA_PRINT_ENUM_VALUE_TO_OS(os, OperandType::kRegister);
    GEMATRIA_PRINT_ENUM_VALUE_TO_OS(os, OperandType::kImmediateValue);
    GEMATRIA_PRINT_ENUM_VALUE_TO_OS(os, OperandType::kFpImmediateValue);
    GEMATRIA_PRINT_ENUM_VALUE_TO_OS(os, OperandType::kAddress);
    GEMATRIA_PRINT_ENUM_VALUE_TO_OS(os, OperandType::kMemory);
    GEMATRIA_PRINT_ENUM_VALUE_TO_OS(os, OperandType::KVirtualRegister);
  }
  return os;
}

#undef GEMATRIA_PRINT_ENUM_VALUE_TO_OS

bool AddressTuple::operator==(const AddressTuple& other) const {
  const auto as_tuple = [](const AddressTuple& address) {
    return std::tie(address.base_register, address.displacement,
                    address.index_register, address.scaling,
                    address.segment_register);
  };
  return as_tuple(*this) == as_tuple(other);
}

std::string AddressTuple::ToString() const {
  std::stringstream buffer;
  buffer << "AddressTuple(";
  if (!base_register.empty()) {
    buffer << "base_register='" << base_register << "', ";
  }
  if (displacement != 0) {
    buffer << "displacement=" << displacement << ", ";
  }
  if (!index_register.empty()) {
    buffer << "index_Register='" << index_register << "', ";
  }
  if (!index_register.empty() || scaling != 0) {
    buffer << "scaling=" << scaling << ", ";
  }
  if (!segment_register.empty()) {
    buffer << "segment_register='" << segment_register << "', ";
  }
  // If we added any keyword args to the buffer, drop the last two characters
  // (a comma and a space). This is not strictly necessary, but it looks better.
  auto msg = buffer.str();
  assert(msg.size() >= 2);
  if (msg.back() == ' ') msg.resize(msg.size() - 2);
  msg.push_back(')');
  return msg;
}

std::ostream& operator<<(std::ostream& os, const AddressTuple& address_tuple) {
  os << address_tuple.ToString();
  return os;
}

bool InstructionOperand::operator==(const InstructionOperand& other) const {
  if (type() != other.type()) return false;
  switch (type()) {
    case OperandType::kUnknown:
      return true;
    case OperandType::kRegister:
      return register_name() == other.register_name();
    case OperandType::kImmediateValue:
      return immediate_value() == other.immediate_value();
    case OperandType::kFpImmediateValue:
      return fp_immediate_value() == other.fp_immediate_value();
    case OperandType::kAddress:
      return address() == other.address();
    case OperandType::kMemory:
      return alias_group_id() == other.alias_group_id();
    case OperandType::KVirtualRegister:
      return register_name() == other.register_name() && size() == other.size();
  }
}

InstructionOperand InstructionOperand::VirtualRegister(
    const std::string register_name, size_t size) {
  InstructionOperand result;
  result.type_ = OperandType::KVirtualRegister;
  result.register_name_ = std::move(register_name);
  result.size_ = size;
  return result;
}

InstructionOperand InstructionOperand::Register(
    const std::string register_name) {
  InstructionOperand result;
  result.type_ = OperandType::kRegister;
  result.register_name_ = std::move(register_name);
  return result;
}

InstructionOperand InstructionOperand::ImmediateValue(
    uint64_t immediate_value) {
  InstructionOperand result;
  result.type_ = OperandType::kImmediateValue;
  result.immediate_value_ = immediate_value;
  return result;
}

InstructionOperand InstructionOperand::FpImmediateValue(
    double fp_immediate_value) {
  InstructionOperand result;
  result.type_ = OperandType::kFpImmediateValue;
  result.fp_immediate_value_ = fp_immediate_value;
  return result;
}

InstructionOperand InstructionOperand::Address(AddressTuple address_tuple) {
  InstructionOperand result;
  result.type_ = OperandType::kAddress;
  result.address_ = std::move(address_tuple);
  return result;
}

InstructionOperand InstructionOperand::Address(std::string base_register,
                                               int64_t displacement,
                                               std::string index_register,
                                               int scaling,
                                               std::string segment_register) {
  InstructionOperand result;
  result.type_ = OperandType::kAddress;
  result.address_.base_register = std::move(base_register);
  result.address_.index_register = std::move(index_register);
  result.address_.displacement = displacement;
  result.address_.scaling = scaling;
  result.address_.segment_register = segment_register;
  return result;
}

InstructionOperand InstructionOperand::MemoryLocation(int alias_group_id) {
  InstructionOperand result;
  result.type_ = OperandType::kMemory;
  result.alias_group_id_ = alias_group_id;
  return result;
}

void InstructionOperand::AddTokensToList(
    std::vector<std::string>& tokens) const {
  switch (type()) {
    case OperandType::kUnknown:
      break;
    case OperandType::kRegister:
      tokens.push_back(register_name());
      break;
    case OperandType::kImmediateValue:
      tokens.emplace_back(kImmediateToken);
      break;
    case OperandType::kFpImmediateValue:
      tokens.emplace_back(kImmediateToken);
      break;
    case OperandType::kAddress:
      tokens.emplace_back(kAddressToken);
      tokens.emplace_back(address().base_register.empty()
                              ? kNoRegisterToken
                              : address().base_register);
      tokens.emplace_back(address().index_register.empty()
                              ? kNoRegisterToken
                              : address().index_register);
      if (!address().segment_register.empty()) {
        tokens.push_back(address().segment_register);
      }
      if (address().displacement != 0) {
        tokens.emplace_back(kDisplacementToken);
      }
      break;
    case OperandType::kMemory:
      tokens.emplace_back(kMemoryToken);
      break;
    default:
      break;
  }
}

std::vector<std::string> InstructionOperand::AsTokenList() const {
  std::vector<std::string> tokens;
  AddTokensToList(tokens);
  return tokens;
}

std::string InstructionOperand::ToString() const {
  std::stringstream buffer;
  buffer << "InstructionOperand";
  switch (type()) {
    case OperandType::kUnknown:
      buffer << "()";
      break;
    case OperandType::kRegister:
      buffer << ".from_register('" << register_name() << "')";
      break;
    case OperandType::kImmediateValue:
      buffer << ".from_immediate_value(" << immediate_value() << ")";
      break;
    case OperandType::kFpImmediateValue:
      buffer << ".from_fp_immediate_value(" << fp_immediate_value() << ")";
      break;
    case OperandType::kAddress:
      buffer << ".from_address(" << address().ToString() << ")";
      break;
    case OperandType::kMemory:
      buffer << ".from_memory(" << alias_group_id() << ")";
      break;
  }
  return buffer.str();
}

std::ostream& operator<<(std::ostream& os, const InstructionOperand& operand) {
  os << operand.ToString();
  return os;
}

Instruction::Instruction(
    std::string mnemonic, std::string llvm_mnemonic,
    std::vector<std::string> prefixes,
    std::vector<InstructionOperand> input_operands,
    std::vector<InstructionOperand> implicit_input_operands,
    std::vector<InstructionOperand> output_operands,
    std::vector<InstructionOperand> implicit_output_operands)
    : mnemonic(std::move(mnemonic)),
      llvm_mnemonic(std::move(llvm_mnemonic)),
      prefixes(std::move(prefixes)),
      input_operands(std::move(input_operands)),
      implicit_input_operands(std::move(implicit_input_operands)),
      output_operands(std::move(output_operands)),
      implicit_output_operands(std::move(implicit_output_operands)) {}

bool Instruction::operator==(const Instruction& other) const {
  const auto as_tuple = [](const Instruction& instruction) {
    return std::tie(
        instruction.mnemonic, instruction.llvm_mnemonic, instruction.prefixes,
        instruction.input_operands, instruction.implicit_input_operands,
        instruction.output_operands, instruction.implicit_output_operands);
  };
  return as_tuple(*this) == as_tuple(other);
}

std::string Instruction::ToString() const {
  std::stringstream buffer;
  buffer << "Instruction(";
  if (!mnemonic.empty()) {
    buffer << "mnemonic='" << mnemonic << "', ";
  }
  if (!llvm_mnemonic.empty()) {
    buffer << "llvm_mnemonic='" << llvm_mnemonic << "', ";
  }
  if (!prefixes.empty()) {
    buffer << "prefixes=(";
    for (const std::string& prefix : prefixes) {
      buffer << "'" << prefix << "', ";
    }
    // Pop only the trailing space. For simplicity, we leave the trailing comma
    // which is required in case there is only one element.
    buffer.seekp(-1, std::ios_base::end);
    buffer << "), ";
  }

  auto add_operand_list = [&buffer](
                              std::string_view name,
                              const std::vector<InstructionOperand>& operands) {
    if (operands.empty()) return;
    buffer << name << "=(";
    for (const InstructionOperand& operand : operands) {
      buffer << operand.ToString() << ", ";
    }
    // Pop only the trailing space. For simplicity, we leave the trailing comma
    // which is required in case there is only one element.
    buffer.seekp(-1, std::ios_base::end);
    buffer << "), ";
  };
  add_operand_list("input_operands", input_operands);
  add_operand_list("implicit_input_operands", implicit_input_operands);
  add_operand_list("output_operands", output_operands);
  add_operand_list("implicit_output_operands", implicit_output_operands);

  auto msg = buffer.str();
  assert(msg.size() >= 2);
  if (msg.back() == ' ') msg.resize(msg.size() - 2);
  msg.push_back(')');
  return msg;
}

void Instruction::AddTokensToList(std::vector<std::string>& tokens) const {
  for (const std::string& prefix : prefixes) tokens.push_back(prefix);
  tokens.push_back(mnemonic);
  tokens.emplace_back(kDelimiterToken);
  for (const auto& operand : output_operands) {
    operand.AddTokensToList(tokens);
  }
  for (const auto& operand : implicit_output_operands) {
    operand.AddTokensToList(tokens);
  }
  tokens.emplace_back(kDelimiterToken);
  for (const auto& operand : input_operands) {
    operand.AddTokensToList(tokens);
  }
  for (const auto& operand : implicit_input_operands) {
    operand.AddTokensToList(tokens);
  }
  tokens.emplace_back(kDelimiterToken);
}

std::vector<std::string> Instruction::AsTokenList() const {
  std::vector<std::string> tokens;
  AddTokensToList(tokens);
  return tokens;
}

std::ostream& operator<<(std::ostream& os, const Instruction& instruction) {
  os << instruction.ToString();
  return os;
}

BasicBlock::BasicBlock(std::vector<Instruction> instructions)
    : instructions(std::move(instructions)) {}

bool BasicBlock::operator==(const BasicBlock& other) const {
  return instructions == other.instructions;
}

std::string BasicBlock::ToString() const {
  std::string buffer = "BasicBlock(";
  if (!instructions.empty()) {
    buffer += "instructions=InstructionList((";
    for (const Instruction& instruction : instructions) {
      buffer += instruction.ToString();
      buffer += ", ";
    }
    if (buffer.back() == ' ') buffer.pop_back();
    buffer += "))";
  }
  buffer.push_back(')');
  return buffer;
}

std::ostream& operator<<(std::ostream& os, const BasicBlock& block) {
  os << block.ToString();
  return os;
}

}  // namespace gematria
