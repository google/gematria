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
#include "google/protobuf/repeated_field.h"

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

namespace {

std::vector<InstructionOperand> ToVector(
    const google::protobuf::RepeatedPtrField<CanonicalizedOperandProto>&
        protos) {
  std::vector<InstructionOperand> result(protos.size());
  std::transform(protos.begin(), protos.end(), result.begin(),
                 InstructionOperandFromProto);
  return result;
}

}  // namespace

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
      ToVector(proto.implicit_output_operands()));
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
