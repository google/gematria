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

#include "gematria/datasets/json_importer.h"

#include <cstdint>
#include <string_view>

#include "gematria/llvm/canonicalizer.h"
#include "llvm/ADT/ArrayRef.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11_abseil/import_status_module.h"
#include "pybind11_abseil/status_casters.h"
#include "pybind11_protobuf/native_proto_caster.h"

namespace gematria {

namespace py = ::pybind11;

PYBIND11_MODULE(json_importer, m) {
  m.doc() = "Support code for importing data from a JSON format.";

  py::google::ImportStatusModule();

  py::class_<JSONImporter>(m, "JsonImporter")
      .def(  //
          py::init<const Canonicalizer* /* canonicalizer */>(),
          py::arg("canonicalizer"),
          R"(Initializes a new JSON importer for a given architecture.

          Args:
            canonicalizer: The canonicalizer used to disassemble instructions
              and convert them to the Gematria proto representation.)")
      .def(  //
          "basic_block_with_throughput_proto_from_json_object",
          &JSONImporter::ParseJSON, py::arg("source_name"),
          py::arg("json_string"), py::arg("throughput_scaling") = 1.0,
          py::arg("base_address") = uint64_t{0},
          R"(Creates a BasicBlockWithThroughputProto from a JSON object.

          Takes a JSON string in the format:
          {
            "machine_code_hex": <machine_code_hex>,
            "instruction_annotations": [
              { "name": <annotation_type_name>,
                "values": [<annotation_value>, ...] },
              ...
            ],
            "throughput": <throughput>
          }
          where <machine_code_hex> is a hex string holding the machine code that
          the basic block is comprised of, <annotation_type_name> is a string
          holding the name of the type of annotation represented by this element
          of "instruction_annotations", <annotation_value> is a numeric value
          corresponding to the annotation type for the instruction sharing the
          same index, and <throughput> is the numeric inverse throughput of the
          basic block. 

          Args:
            source_name: The name of the throughput source used in the output
              proto.
            json_string: The JSON object representing a basic block.
            throughput_column_index: The index of the column in the CSV
              containing the throughput in cycles.
            throughput_scaling: An optional scaling applied to {throughput}.
            base_address: The address of the first instruction of the basic
              block.

          Returns:
            A BasicBlockWithThroughputProto that contains the basic block
            extracted from {machine_code}, and one throughput information based
            on {throughput}.

          Raises:
            StatusNotOk: When parsing the JSON string or extracting data from the
              machine code fails.)");
}

}  // namespace gematria
