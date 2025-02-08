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
#include "gematria/proto/annotation.pb.h"
#include "gematria/proto/canonicalized_instruction.pb.h"
#include "google/protobuf/repeated_ptr_field.h"

namespace gematria {

namespace {

template <typename Object, typename Proto, typename Convertor>
std::vector<Object> ToVector(
    const google::protobuf::RepeatedPtrField<Proto>& protos,
    Convertor object_from_proto) {
  std::vector<Object> result(std::size(protos));
  std::transform(std::begin(protos), std::end(protos), std::begin(result),
                 object_from_proto);
  return result;
}

template <typename Object, typename Proto, typename Convertor>
void ToRepeatedPtrField(
    const std::vector<Object>& objects,
    google::protobuf::RepeatedPtrField<Proto>* repeated_field,
    Convertor proto_from_object) {
  repeated_field->Reserve(std::size(objects));
  std::transform(std::begin(objects), std::end(objects),
                 google::protobuf::RepeatedFieldBackInserter(repeated_field),
                 proto_from_object);
}

}  // namespace

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

Annotation AnnotationFromProto(const AnnotationProto& proto) {
  return Annotation(
      /* name = */ proto.name(),
      /* value = */ proto.value());
}

AnnotationProto ProtoFromAnnotation(const Annotation& annotation) {
  AnnotationProto proto;
  proto.set_name(annotation.name);
  proto.set_value(annotation.value);
  return proto;
}

Instruction InstructionFromProto(const CanonicalizedInstructionProto& proto) {
  return Instruction(
      /* mnemonic = */ proto.mnemonic(),
      /* llvm_mnemonic = */ proto.llvm_mnemonic(),
      /* prefixes = */
      std::vector<std::string>(proto.prefixes().begin(),
                               proto.prefixes().end()),
      /* input_operands = */
      ToVector<InstructionOperand>(proto.input_operands(),
                                   InstructionOperandFromProto),
      /* implicit_input_operands = */
      ToVector<InstructionOperand>(proto.implicit_input_operands(),
                                   InstructionOperandFromProto),
      /* output_operands = */
      ToVector<InstructionOperand>(proto.output_operands(),
                                   InstructionOperandFromProto),
      /* implicit_output_operands = */
      ToVector<InstructionOperand>(proto.implicit_output_operands(),
                                   InstructionOperandFromProto),
      /* instruction_annotations = */
      ToVector<Annotation>(proto.instruction_annotations(),
                           AnnotationFromProto));
}

CanonicalizedInstructionProto ProtoFromInstruction(
    const Instruction& instruction) {
  CanonicalizedInstructionProto proto;
  proto.set_mnemonic(instruction.mnemonic);
  proto.set_llvm_mnemonic(instruction.llvm_mnemonic);
  proto.mutable_prefixes()->Assign(instruction.prefixes.begin(),
                                   instruction.prefixes.end());
  ToRepeatedPtrField(instruction.input_operands, proto.mutable_input_operands(),
                     ProtoFromInstructionOperand);
  ToRepeatedPtrField(instruction.implicit_input_operands,
                     proto.mutable_implicit_input_operands(),
                     ProtoFromInstructionOperand);
  ToRepeatedPtrField(instruction.output_operands,
                     proto.mutable_output_operands(),
                     ProtoFromInstructionOperand);
  ToRepeatedPtrField(instruction.implicit_output_operands,
                     proto.mutable_implicit_output_operands(),
                     ProtoFromInstructionOperand);
  ToRepeatedPtrField(instruction.instruction_annotations,
                     proto.mutable_instruction_annotations(),
                     ProtoFromAnnotation);
  return proto;
}

BasicBlock BasicBlockFromProto(const BasicBlockProto& proto) {
  return BasicBlock(
      /* instructions = */
      ToVector<Instruction>(proto.canonicalized_instructions(),
                            InstructionFromProto),
      /* preceding_context = */
      ToVector<Instruction>(proto.canonicalized_preceding_context(),
                            InstructionFromProto),
      /* following_context = */
      ToVector<Instruction>(proto.canonicalized_following_context(),
                            InstructionFromProto));
}

}  // namespace gematria
