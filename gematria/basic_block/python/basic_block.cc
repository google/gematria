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

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
PYBIND11_MAKE_OPAQUE(std::vector<gematria::InstructionOperand>);
PYBIND11_MAKE_OPAQUE(std::vector<gematria::Instruction>);

namespace gematria {

namespace py = ::pybind11;

// Safe version of applying a getter to InstructionOperand for cases when the
// getter returns a value (neither a reference nor a pointer).
// - when operand.type() == expected_type, returns the value returned by
//   getter_member_ptr when applied to `operand`.
// - Otherwise, returns std::nullopt; this is converted by pybind11 to None.
template <OperandType expected_type, auto getter_member_ptr,
          typename ResultType = decltype((
              InstructionOperand::ImmediateValue(0).*getter_member_ptr)())>
std::enable_if_t<!std::is_reference_v<ResultType> &&
                     !std::is_pointer_v<ResultType>,
                 std::optional<ResultType>>
InstructionOperandPropertyOrNone(const InstructionOperand& operand) {
  if (operand.type() != expected_type) {
    return std::nullopt;
  }
  return std::make_optional((operand.*getter_member_ptr)());
}

// Safe version of reading a field of InstructionOperand for cases when the
// getter returns a reference. Preserves const-ness of the returned value.
// - when operand.type() == expected_type, returns a pointer to the returned
//   object.
// - otherwise, returns nullptr; this is converted by pybind11 to None.
// This wrapper can be used together with
// py::return_value_policy::reference_internal to avoid unnecessary copying of
// the address tuple.
template <OperandType expected_type, auto getter_member_ptr,
          typename ResultType = decltype((
              InstructionOperand::ImmediateValue(0).*getter_member_ptr)())>
std::enable_if_t<std::is_lvalue_reference_v<ResultType>,
                 std::add_pointer_t<ResultType>>
InstructionOperandPropertyOrNone(const InstructionOperand& operand) {
  if (operand.type() != expected_type) {
    return nullptr;
  }
  return &(operand.*getter_member_ptr)();
}

// Safe version of reading a field of InstructionOperand for cases when the
// getter returns a reference. Preserves const-ness of the returned value.
// - when operand.type() == expected_type1 or operand.type() == expected_type2, returns a pointer to the returned
//   object.
// - otherwise, returns nullptr; this is converted by pybind11 to None.
// This wrapper can be used together with
// py::return_value_policy::reference_internal to avoid unnecessary copying of
// the address tuple.
template <OperandType expected_type1, OperandType expected_type2, auto getter_member_ptr,
          typename ResultType = decltype((
              InstructionOperand::ImmediateValue(0).*getter_member_ptr)())>
std::enable_if_t<std::is_lvalue_reference_v<ResultType>,
                 std::add_pointer_t<ResultType>>
InstructionOperandVregPropertyOrNone(const InstructionOperand& operand) {
  if (operand.type() != expected_type1 || operand.type() != expected_type2 ) {
    return nullptr;
  }
  return &(operand.*getter_member_ptr)();
}

PYBIND11_MODULE(basic_block, m) {
  m.doc() = "Data structures representing instructions and basic blocks.";

  // Use bound versions of the two vector types. This makes changes done in
  // Python code propagate to C++ code.
  py::bind_vector<std::vector<std::string>>(m, "StringList");

  py::enum_<OperandType>(m, "OperandType", R"(
      The type of the operands used in the basic blocks.

      Values:
        REGISTER: The operand is a register.
        IMMEDIATE_VALUE: The operand is an integer immediate value. This
          immediate value can have up to 64-bits.
        FP_IMMEDIATE_VALUE: The operand is a floating-point immediate value.
        ADDRESS: The operand is an address computation.
        MEMORY: The operand is a location in the memory.)")
      .value("UNKNOWN", OperandType::kUnknown)
      .value("REGISTER", OperandType::kRegister)
      .value("IMMEDIATE_VALUE", OperandType::kImmediateValue)
      .value("FP_IMMEDIATE_VALUE", OperandType::kFpImmediateValue)
      .value("ADDRESS", OperandType::kAddress)
      .value("MEMORY", OperandType::kMemory)
      .value("VIRTUAL_REGISTER", OperandType::kVirtualRegister);

  py::class_<AddressTuple> address_tuple(m, "AddressTuple");
  address_tuple
      .def(py::init<std::string /* base_register */, int64_t /* displacement */,
                    std::string /* index_register */, int /* scaling */,
                    std::string /* segment_register */>(),
           py::arg("base_register") = std::string(),
           py::arg("displacement") = 0,
           py::arg("index_register") = std::string(), py::arg("scaling") = 0,
           py::arg("segment_register") = std::string())
      .def("__repr__", &AddressTuple::ToString)
      .def("__eq__", &AddressTuple::operator==)
      .def("__copy__",
           [](const AddressTuple& address_tuple) {
             return AddressTuple(address_tuple);
           })
      .def(
          "__deepcopy__",
          [](const AddressTuple& address_tuple, py::dict) {
            return AddressTuple(address_tuple);
          },
          py::arg("memo"))
      .def_readonly("base_register", &AddressTuple::base_register)
      .def_readonly("index_register", &AddressTuple::index_register)
      .def_readonly("displacement", &AddressTuple::displacement)
      .def_readonly("scaling", &AddressTuple::scaling)
      .def_readonly("segment_register", &AddressTuple::segment_register);

  py::class_<InstructionOperand>(m, "InstructionOperand")
      .def_static("from_register", &InstructionOperand::Register,
                  py::arg("register_name"))
      .def_static("from_immediate_value", &InstructionOperand::ImmediateValue,
                  py::arg("immediate_value"))
      .def_static("from_fp_immediate_value",
                  &InstructionOperand::FpImmediateValue,
                  py::arg("fp_immediate_value"))
      .def_static("from_virtual_register",
                  &InstructionOperand::VirtualRegister,
                  py::arg("register_name"), py::arg("size"), py::arg("interfered_registers"))
      .def_static<InstructionOperand (*)(
          std::string /* base_register */, int64_t /* displacement */,
          std::string /* index_register */, int /* scaling */,
          std::string /* segment_register */, int /* base_register_size */,
          int /* index_register_size */, int /* segment_register_size */)>(
          "from_address", &InstructionOperand::Address,
          py::arg("base_register") = std::string(), py::arg("displacement") = 0,
          py::arg("index_register") = std::string(), py::arg("scaling") = 0,
          py::arg("segment_register") = std::string(), py::arg("base_register_size") = 64,
          py::arg("index_register_size") = 64, py::arg("segment_register_size") = 64)
      .def_static<InstructionOperand (*)(AddressTuple)>(
          "from_address", &InstructionOperand::Address,
          py::arg("address_tuple"))
      .def_static("from_memory", &InstructionOperand::MemoryLocation,
                  py::arg("alias_group_id"))
      .def("__eq__", &InstructionOperand::operator==)
      .def("__repr__", &InstructionOperand::ToString)
      .def("__str__", &InstructionOperand::ToString)
      .def("__copy__",
           [](const InstructionOperand& operand) {
             return InstructionOperand(operand);
           })
      .def(
          "__deepcopy__",
          [](const InstructionOperand& operand, py::dict) {
            return InstructionOperand(operand);
          },
          py::arg("memo"))
      .def("as_token_list", &InstructionOperand::AsTokenList)
      .def_property_readonly("type", &InstructionOperand::type)
      .def_property_readonly(
          "register_name",InstructionOperandVregPropertyOrNone<OperandType::kRegister, OperandType::kVirtualRegister, &InstructionOperand::register_name>)
      .def_property_readonly(
          "size", &InstructionOperand::size)
      .def_property_readonly("immediate_value",
                             InstructionOperandPropertyOrNone<
                                 OperandType::kImmediateValue,
                                 &InstructionOperand::immediate_value>)
      .def_property_readonly("fp_immediate_value",
                             InstructionOperandPropertyOrNone<
                                 OperandType::kFpImmediateValue,
                                 &InstructionOperand::fp_immediate_value>)
      .def_property_readonly(
          "address",
          InstructionOperandPropertyOrNone<OperandType::kAddress,
                                           &InstructionOperand::address>,
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "alias_group_id",
          InstructionOperandPropertyOrNone<
              OperandType::kMemory, &InstructionOperand::alias_group_id>);

  py::bind_vector<std::vector<InstructionOperand>>(m, "InstructionOperandList");

  py::class_<Instruction>(m, "Instruction")
      .def(
          py::init<
              std::string /* mnemonic */, std::string /* llvm_mnemonic */,
              std::vector<std::string> /* prefixes */,
              std::vector<InstructionOperand> /* input_operands */,
              std::vector<InstructionOperand> /* implicit_input_operands */,
              std::vector<InstructionOperand> /* output_oeprands */,
              std::vector<InstructionOperand> /* implicit_output_operands */>(),
          py::arg("mnemonic") = std::string(),
          py::arg("llvm_mnemonic") = std::string(),
          py::arg("prefixes") = std::vector<std::string>(),
          py::arg("input_operands") = std::vector<InstructionOperand>(),
          py::arg("implicit_input_operands") =
              std::vector<InstructionOperand>(),
          py::arg("output_operands") = std::vector<InstructionOperand>(),
          py::arg("implicit_output_operands") =
              std::vector<InstructionOperand>())
      .def("__str__", &Instruction::ToString)
      .def("__repr__", &Instruction::ToString)
      .def("__eq__", &Instruction::operator==)
      .def("__copy__",
           [](const Instruction& instruction) {
             return Instruction(instruction);
           })
      .def(
          "__deepcopy__",
          [](const Instruction& instruction, py::dict) {
            return Instruction(instruction);
          },
          py::arg("memo"))
      .def("as_token_list", &Instruction::AsTokenList)
      .def_readwrite("mnemonic", &Instruction::mnemonic)
      .def_readwrite("prefixes", &Instruction::prefixes)
      .def_readwrite("llvm_mnemonic", &Instruction::llvm_mnemonic)
      .def_readwrite("input_operands", &Instruction::input_operands)
      .def_readwrite("implicit_input_operands",
                     &Instruction::implicit_input_operands)
      .def_readwrite("output_operands", &Instruction::output_operands)
      .def_readwrite("implicit_output_operands",
                     &Instruction::implicit_output_operands);

  py::bind_vector<std::vector<Instruction>>(m, "InstructionList");

  py::class_<BasicBlock> basic_block(m, "BasicBlock");
  basic_block
      .def(py::init<std::vector<Instruction> /* instructions */>(),
           py::arg("instructions") = std::vector<Instruction>())
      .def_readwrite("instructions", &BasicBlock::instructions)
      .def("__repr__", &BasicBlock::ToString)
      .def("__str__", &BasicBlock::ToString)
      .def("__eq__", &BasicBlock::operator==)
      .def("__copy__",
           [](const BasicBlock& block) { return BasicBlock(block); })
      .def(
          "__deepcopy__",
          [](const BasicBlock& block, py::dict) { return BasicBlock(block); },
          py::arg("memo"));
}

}  // namespace gematria
