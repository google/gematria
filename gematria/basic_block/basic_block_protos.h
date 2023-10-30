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

// Functions for working with basic block protos. These functions are
// intentionally kept in a separate library to isolate the proto dependences to
// the smallest scope possible.

#ifndef GEMATRIA_BASIC_BLOCK_BASIC_BLOCK_PROTOS_H_
#define GEMATRIA_BASIC_BLOCK_BASIC_BLOCK_PROTOS_H_

#include "gematria/basic_block/basic_block.h"
#include "gematria/proto/basic_block.pb.h"
#include "gematria/proto/canonicalized_instruction.pb.h"

namespace gematria {

// Creates an address tuple data structure from a proto.
AddressTuple AddressTupleFromProto(
    const CanonicalizedOperandProto::AddressTuple& proto);

// Creates a proto representing the given address tuple.
CanonicalizedOperandProto::AddressTuple ProtoFromAddressTuple(
    const AddressTuple& address_tuple);

// Creates an instruction operand data structure from a proto.
InstructionOperand InstructionOperandFromProto(
    const CanonicalizedOperandProto& proto);

// Creates a proto representing the given instruction operand.
CanonicalizedOperandProto ProtoFromInstructionOperand(
    const InstructionOperand& operand);

// Creates a runtime annotation data structure from a proto.
RuntimeAnnotation RuntimeAnnotationFromProto(
    const CanonicalizedInstructionProto::RuntimeAnnotation& proto);

// Creates a proto representing the given runtime annotation.
CanonicalizedInstructionProto::RuntimeAnnotation ProtoFromRuntimeAnnotation(
    const RuntimeAnnotation& runtime_annotation);

// Creates an instruction data structure from a proto.
Instruction InstructionFromProto(const CanonicalizedInstructionProto& proto);

// Creates a proto representing the given instruction.
CanonicalizedInstructionProto ProtoFromInstruction(
    const Instruction& instruction);

// Creates a basic block data structure from a proto.
BasicBlock BasicBlockFromProto(const BasicBlockProto& proto);

}  // namespace gematria

#endif  // GEMATRIA_BASIC_BLOCK_BASIC_BLOCK_PROTOS_H_
