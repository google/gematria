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

// Contains definitions of data structures that are used to represent
// instructions and basic blocks in the project code.
//
// This library is designed to have minimal dependencies beyond the standard
// library. The data structures defined in this library are shared between C++
// and Python via https://github.com/pybind/pybind11 bindings.

#ifndef GEMATRIA_BASIC_BLOCK_BASIC_BLOCK_H_
#define GEMATRIA_BASIC_BLOCK_BASIC_BLOCK_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gematria {

// Tokens used for instruction canonicalization in Gematria. The values used
// here must be kept in sync with the values defined in
// gematria/basic_block/python/tokens.py.
inline constexpr std::string_view kDelimiterToken = "_D_";
inline constexpr std::string_view kImmediateToken = "_IMMEDIATE_";
inline constexpr std::string_view kAddressToken = "_ADDRESS_";
inline constexpr std::string_view kMemoryToken = "_MEMORY_";
inline constexpr std::string_view kNoRegisterToken = "_NO_REGISTER_";
inline constexpr std::string_view kDisplacementToken = "_DISPLACEMENT_";

// The type of an operand of an instruction.
enum class OperandType {
  // The type of the operand is not known/the operand was not correctly
  // initialized.
  kUnknown,

  // The operand is a register of the CPU.
  kRegister,

  // The operand is an integer immediate value.
  kImmediateValue,

  // The operand is a floating-point immediate value.
  kFpImmediateValue,

  // The operand is a memory address. Note that this operand type represents the
  // computation of the address, not a memory access at this address. For
  // example the LEA (load effective address) would have an operand of this type
  // even though it does not access memory. Instructions that use address
  // computation and access the memory have also an operand of type kMemory.
  kAddress,

  // The operand is a memory access. Instructions with this operand often have
  // also an operand of type kAddress.
  kMemory,
};

std::ostream& operator<<(std::ostream& os, OperandType operand_type);

// Represents inputs to address computation of an instruction.
struct AddressTuple {
  AddressTuple() {}
  AddressTuple(const AddressTuple&) = default;
  AddressTuple(AddressTuple&&) = default;
  AddressTuple(std::string base_register, int64_t displacement,
               std::string index_register, int scaling,
               std::string segment_register)
      : base_register(std::move(base_register)),
        displacement(displacement),
        index_register(std::move(index_register)),
        scaling(scaling),
        segment_register(std::move(segment_register)) {}

  AddressTuple& operator=(const AddressTuple&) = default;
  AddressTuple& operator=(AddressTuple&&) = default;

  bool operator==(const AddressTuple& other) const;
  bool operator!=(const AddressTuple& other) const { return !(*this == other); }

  // Returns a human-readable string representation of the address tuple. This
  // representation corresponds to Python code that creates the same object,
  // e.g. "AddressTuple(base_register='RAX', displacement=16,
  //                    index_register='RBX', scaling=3,
  //                    segment_register='FS')" (without the indentation).
  // Only fields that are used appear in the string representation.
  std::string ToString() const;

  // The name of the base register of the address. When empty, base register
  // is not used in the computation.
  std::string base_register;

  // An absolute displacement (offset) of the address. When zero, a displacement
  // is not used.
  int64_t displacement = 0;

  // The name of the index register of the address. When empty, index register
  // is not used in the computation.
  std::string index_register;

  // The scaling factor of the index register. The interpretation depends on the
  // used architecture; on x86-64 it is the log_2 of the scaling factor applied
  // to the value of index register. Used only when index_register is non-empty.
  int scaling = 0;

  // The name of the segment register. When empty, the default segment register
  // for the instruction is used.
  std::string segment_register;
};

std::ostream& operator<<(std::ostream& os, const AddressTuple& address_tuple);

// Represents a single operand of an instruction. Only the getters related to
// the represented operand type may be used; use of methods that are not valid
// for the represented operand type will lead to undefined behavior.
class InstructionOperand {
 public:
  // Creates an operand of type kUnknown.
  InstructionOperand() {}

  // Operands are movable and assignable.
  InstructionOperand(const InstructionOperand&) = default;
  InstructionOperand(InstructionOperand&&) = default;

  InstructionOperand& operator=(const InstructionOperand&) = default;
  InstructionOperand& operator=(InstructionOperand&&) = default;

  // The operands must be created through one of the factory functions.
  static InstructionOperand Register(std::string register_name);
  static InstructionOperand ImmediateValue(uint64_t immediate_value);
  static InstructionOperand FpImmediateValue(double fp_immediate_value);
  static InstructionOperand Address(AddressTuple address_tuple);
  static InstructionOperand Address(std::string base_register,
                                    int64_t displacement,
                                    std::string index_register, int scaling,
                                    std::string segment_register);
  static InstructionOperand MemoryLocation(int alias_group_id);

  bool operator==(const InstructionOperand&) const;
  bool operator!=(const InstructionOperand& other) const {
    return !(*this == other);
  }

  // Adds tokens of the instruction to a list of tokens.
  void AddTokensToList(std::vector<std::string>& tokens) const;

  // Returns the list of tokens representing this instruction.
  std::vector<std::string> AsTokenList() const;

  // Returns a human-readable representation of the operand.
  //
  // This method implements the __str__() and __repr__() methods in the Python
  // version of the class, and the string representation is Python code that
  // creates the object, e.g. "InstructionOperand.from_register('RAX')".
  std::string ToString() const;

  // Returns the type of the operand. Valid for all operand types including
  // kUnknown.
  OperandType type() const { return type_; }

  // Returns the name of the register. Valid only when type() is kRegister.
  const std::string& register_name() const {
    assert(type_ == OperandType::kRegister);
    return register_name_;
  }

  // Returns the immediate value in the operand. Valid only when type() is
  // kImmediateValue.
  uint64_t immediate_value() const {
    assert(type_ == OperandType::kImmediateValue);
    return immediate_value_;
  }

  // Returns the floating point immediate value in the operand. Valid only when
  // type() is kFpImmediateValue.
  double fp_immediate_value() const {
    assert(type_ == OperandType::kFpImmediateValue);
    return fp_immediate_value_;
  }

  // Returns the address computation data structure in the operand. Valid only
  // when type() is kAddress.
  const AddressTuple& address() const {
    assert(type_ == OperandType::kAddress);
    return address_;
  }

  // Returns the alias group ID of the memory access in the operand. Valid only
  // when type() is kMemory.
  int alias_group_id() const {
    assert(type_ == OperandType::kMemory);
    return alias_group_id_;
  }

 private:
  OperandType type_ = OperandType::kUnknown;

  std::string register_name_;
  uint64_t immediate_value_ = 0;
  double fp_immediate_value_ = 0.0;
  AddressTuple address_;
  int alias_group_id_ = 0;
};

std::ostream& operator<<(std::ostream& os, const InstructionOperand& operand);

// Represents a single instruction.
struct Instruction {
  Instruction() {}

  // Initializes all fields of the instruction. Needed for compatibility with
  // the Python code.
  Instruction(std::string mnemonic, std::string llvm_mnemonic,
              std::vector<std::string> prefixes,
              std::vector<InstructionOperand> input_operands,
              std::vector<InstructionOperand> implicit_input_operands,
              std::vector<InstructionOperand> output_operands,
              std::vector<InstructionOperand> implicit_output_operands);

  Instruction(const Instruction&) = default;
  Instruction(Instruction&&) = default;

  Instruction& operator=(const Instruction&) = default;
  Instruction& operator=(Instruction&&) = default;

  bool operator==(const Instruction& other) const;
  bool operator!=(const Instruction& other) const { return !(*this == other); }

  // Returns the list of tokens representing this instruction. The returned list
  // contains the tokens from the assembly representation of the instruction
  // with delimiter tokens separating the instructions and the different types
  // of its operands.
  std::vector<std::string> AsTokenList() const;
  void AddTokensToList(std::vector<std::string>& tokens) const;

  // Returns a human-readable representation of the instruction.
  //
  // This method implements the __str__() and __repr__() methods in the Python
  // version of the class, and the string representation return Python code that
  // creates the object.
  std::string ToString() const;

  // The mnemonic of the instruction used to represent the instruction in the
  // model.
  std::string mnemonic;
  // The LLVM mnemonic of the instruction. Note that the LLVM mnemonics tend to
  // change with LLVM versions, and we do not recommend using it in models.
  std::string llvm_mnemonic;

  // The list of instruction prefixes. These are additional strings that can be
  // added to the mnemonic. In the models, they are typically represented by
  // their own embedding vector.
  std::vector<std::string> prefixes;

  // The list of explicit input operands of the instruction.
  std::vector<InstructionOperand> input_operands;
  // The list of implicit input operands of the instruction. To make the machine
  // learning task easier, we present also the implicit input operands to the
  // ML models explicitly.
  std::vector<InstructionOperand> implicit_input_operands;

  // The list of explicit output operands of the instruction.
  std::vector<InstructionOperand> output_operands;
  // The list of implicit output operands of the instruction. To make the
  // machine learning task easier, we present also the explicit output operands
  // to the ML models explicitly.
  std::vector<InstructionOperand> implicit_output_operands;

  // The address of the instruction.
  uint64_t address = 0;
  // The size of the instruction.
  size_t size = 0;

  // The instruction is valid or not
  bool is_valid = true;
};

std::ostream& operator<<(std::ostream& os, const Instruction& instruction);

// Represents a basic block, i.e. a sequence of instructions.
struct BasicBlock {
  BasicBlock() {}

  // Initializes the basic block from a list of instructions. Needed for
  // compatibility with the Python code.
  explicit BasicBlock(std::vector<Instruction> instructions);

  BasicBlock(const BasicBlock&) = default;
  BasicBlock(BasicBlock&&) = default;

  BasicBlock& operator=(const BasicBlock&) = default;
  BasicBlock& operator=(BasicBlock&&) = default;

  bool operator==(const BasicBlock& other) const;
  bool operator!=(const BasicBlock& other) const { return !(*this == other); }

  // Returns a human-readable representation of the basic block.
  //
  // This method implements the __str__() and __repr__() methods in the Python
  // version of the class, and the string representation return Python code that
  // creates the object.
  std::string ToString() const;

  // The list of instructions in the basic block.
  std::vector<Instruction> instructions;
};

std::ostream& operator<<(std::ostream& os, const BasicBlock& block);

}  // namespace gematria

#endif  // GEMATRIA_BASIC_BLOCK_BASIC_BLOCK_H_
