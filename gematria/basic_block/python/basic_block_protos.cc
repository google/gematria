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

#include "gematria/proto/canonicalized_instruction.pb.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11_protobuf/native_proto_caster.h"

namespace gematria {

namespace py = ::pybind11;

PYBIND11_MODULE(basic_block_protos, m) {
  pybind11_protobuf::ImportNativeProtoCasters();

  m.doc() = "Functions for converting protos to Gematria data structures.";

  m.def("basic_block_from_proto", BasicBlockFromProto, py::arg("proto"));
  m.def("instruction_from_proto", InstructionFromProto, py::arg("proto"));
  m.def("instruction_operand_from_proto", InstructionOperandFromProto,
        py::arg("proto"));
  m.def("address_tuple_from_proto", AddressTupleFromProto, py::arg("proto"));
  m.def("annotation_from_proto", AnnotationFromProto,
        py::arg("proto"));
}

}  // namespace gematria
