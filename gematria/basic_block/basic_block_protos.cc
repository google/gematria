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

#include "gematria/basic_block/basic_block_protos.h"

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include "gematria/basic_block/basic_block.h"
#include "gematria/proto/canonicalized_instruction.pb.h"
#include "google/protobuf/repeated_ptr_field.h"

namespace gematria {

AddressTuple AddressTupleFromProto(
    const CanonicalizedOperandProto::AddressTuple& proto) {
  return AddressTuple(
      /* base_register = */ proto.base_register(),
      /* displacement = */ proto.displacement(),
      /* index_register = */ proto.index_register(),
      /* scaling = */ proto.scaling(),
      /* segment_register = */ proto.segment());
}

CanonicalizedOperandProto::AddressTuple ProtoFromAddressTuple(
    const AddressTuple& address_tuple) {
  CanonicalizedOperandProto::AddressTuple proto;
  proto.set_base_register(address_tuple.base_register);
  proto.set_displacement(address_tuple.displacement);
  proto.set_index_register(address_tuple.index_register);
  proto.set_scaling(address_tuple.scaling);
  proto.set_segment(address_tuple.segment_register);
  return proto;
}

InstructionOperand InstructionOperandFromProto(
    const CanonicalizedOperandProto& proto) {
  switch (proto.operand_case()) {
    case CanonicalizedOperandProto::OPERAND_NOT_SET:
      return InstructionOperand();
    case CanonicalizedOperandProto::kRegisterName:
      return InstructionOperand::Register(proto.register_name());
    case CanonicalizedOperandProto::kImmediateValue:
      return InstructionOperand::ImmediateValue(proto.immediate_value());
    case CanonicalizedOperandProto::kFpImmediateValue:
      return InstructionOperand::FpImmediateValue(proto.fp_immediate_value());
    case CanonicalizedOperandProto::kAddress:
      return InstructionOperand::Address(
          AddressTupleFromProto(proto.address()));
    case CanonicalizedOperandProto::kMemory:
      return InstructionOperand::MemoryLocation(
          proto.memory().alias_group_id());
  }
}

CanonicalizedOperandProto ProtoFromInstructionOperand(
    const InstructionOperand& operand) {
  CanonicalizedOperandProto proto;
  switch (operand.type()) {
    case OperandType::kRegister:
      proto.set_register_name(operand.register_name());
      break;
    case OperandType::kImmediateValue:
      proto.set_immediate_value(operand.immediate_value());
      break;
    case OperandType::kFpImmediateValue:
      proto.set_fp_immediate_value(operand.fp_immediate_value());
      break;
    case OperandType::kAddress:
      *proto.mutable_address() = ProtoFromAddressTuple(operand.address());
      break;
    case OperandType::kMemory:
      proto.mutable_memory()->set_alias_group_id(operand.alias_group_id());
      break;
    case OperandType::kUnknown:
      break;
  }
  return proto;
}

namespace {

std::vector<InstructionOperand> ToVector(
    const google::protobuf::RepeatedPtrField<CanonicalizedOperandProto>&
        protos) {
  std::vector<InstructionOperand> result(protos.size());
  std::transform(protos.begin(), protos.end(), result.begin(),
                 InstructionOperandFromProto);
  return result;
}

void ToRepeatedPtrField(
    const std::vector<InstructionOperand>& operands,
    google::protobuf::RepeatedPtrField<CanonicalizedOperandProto>*
        repeated_field) {
  repeated_field->Reserve(operands.size());
  std::transform(operands.begin(), operands.end(),
                 google::protobuf::RepeatedFieldBackInserter(repeated_field),
                 ProtoFromInstructionOperand);
}

}  // namespace

RuntimeAnnotation RuntimeAnnotationFromProto(
    const CanonicalizedInstructionProto::RuntimeAnnotation& proto) {
  return RuntimeAnnotation(
      /* pmu_event = */ proto.pmu_event(),
      /* value = */ proto.value());
}

CanonicalizedInstructionProto::RuntimeAnnotation ProtoFromRuntimeAnnotation(
    const RuntimeAnnotation& runtime_annotation) {
  CanonicalizedInstructionProto::RuntimeAnnotation proto;
  proto.set_pmu_event(runtime_annotation.pmu_event);
  proto.set_value(runtime_annotation.value);
  return proto;
}

Instruction InstructionFromProto(const CanonicalizedInstructionProto& proto) {
  return Instruction(
      /* mnemonic = */ proto.mnemonic(),
      /* llvm_mnemonic = */ proto.llvm_mnemonic(),
      /* prefixes = */
      std::vector<std::string>(proto.prefixes().begin(),
                               proto.prefixes().end()),
      /* input_operands = */ ToVector(proto.input_operands()),
      /* implicit_input_operands = */ ToVector(proto.implicit_input_operands()),
      /* output_operands = */ ToVector(proto.output_operands()),
      /* implicit_output_operands = */
      ToVector(proto.implicit_output_operands()),
      /* cache_miss_frequency = */
      RuntimeAnnotationFromProto(proto.cache_miss_frequency()));
}

CanonicalizedInstructionProto ProtoFromInstruction(
    const Instruction& instruction) {
  CanonicalizedInstructionProto proto;
  proto.set_mnemonic(instruction.mnemonic);
  proto.set_llvm_mnemonic(instruction.llvm_mnemonic);
  proto.mutable_prefixes()->Assign(instruction.prefixes.begin(),
                                   instruction.prefixes.end());
  ToRepeatedPtrField(instruction.input_operands,
                     proto.mutable_input_operands());
  ToRepeatedPtrField(instruction.implicit_input_operands,
                     proto.mutable_implicit_input_operands());
  ToRepeatedPtrField(instruction.output_operands,
                     proto.mutable_output_operands());
  ToRepeatedPtrField(instruction.implicit_output_operands,
                     proto.mutable_implicit_output_operands());
  return proto;
}

namespace {

std::vector<Instruction> ToVector(
    const google::protobuf::RepeatedPtrField<CanonicalizedInstructionProto>&
        protos) {
  std::vector<Instruction> result(protos.size());
  std::transform(protos.begin(), protos.end(), result.begin(),
                 InstructionFromProto);
  return result;
}

}  // namespace

BasicBlock BasicBlockFromProto(const BasicBlockProto& proto) {
  return BasicBlock(
      /* instructions = */ ToVector(proto.canonicalized_instructions()));
}

}  // namespace gematria
