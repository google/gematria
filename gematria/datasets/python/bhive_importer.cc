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

#include "gematria/datasets/bhive_importer.h"

#include <cstdint>
#include <string_view>

#include "gematria/llvm/canonicalizer.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11_abseil/import_status_module.h"
#include "pybind11_abseil/status_casters.h"
#include "pybind11_protobuf/native_proto_caster.h"

namespace gematria {

namespace py = ::pybind11;

PYBIND11_MODULE(bhive_importer, m) {
  m.doc() = "Support code for importing data from the BHive data set format.";

  py::google::ImportStatusModule();

  py::class_<BHiveImporter>(m, "BHiveImporter")
      .def(  //
          py::init<const Canonicalizer* /* canonicalizer */>(),
          py::arg("canonicalizer"),
          R"(Initializes a new BHive importer for a given architecture.

          Args:
            canonicalizer: The canonicalizer used to disassemble instructions
              and convert them to the Gematria proto representation.)")
      .def(  //
          "basic_block_proto_from_bytes",
          [](BHiveImporter& self, py::bytes machine_code,
             uint64_t base_address) {
            // We need to explicitly "convert" from a Python bytes object to
            // an array of uint8_t.
            std::string_view machine_code_view = machine_code;
            absl::Span<const uint8_t> machine_code_bytes(
                reinterpret_cast<const uint8_t*>(machine_code_view.data()),
                machine_code_view.size());
            return self.BasicBlockProtoFromMachineCode(machine_code_bytes,
                                                       base_address);
          },
          py::arg("machine_code"), py::arg("base_address") = uint64_t{0},
          R"(Creates a BasicBlockProto from raw machine code.

          Disassembles `machine_code` and creates a new BasicBlockProto that
          contains all instructions extracted from `machine_code`. Uses
          `base_address` as the address of the first instruction.

          Args:
            machine_code: A `bytes` object that contains the machine code.
            base_address: The address of the first instruction of the basic
              block.

          Returns:
            A BasicBlockProto containing the instructions from `machine_code`.

          Raises:
            StatusNotOk: When extracting instructions from the machine code
              fails.)")
      .def(  //
          "basic_block_proto_from_hex",
          &BHiveImporter::BasicBlockProtoFromMachineCodeHex,
          py::arg("machine_code_hex"), py::arg("base_address") = uint64_t{0},
          R"(Creates a BasicBlockProto from machine code in hex string.

          Similar to `basic_block_proto_from_bytes` but the machine code is
          represented as a string of hex digits.

          Args:
            machine_code_hex: The machine code of the basic block represented as
              a string of hex digits, with two digits per byte and no separators
              between them.
            base_address: The address of the first instruction of the basic
              block.

          Returns:
            A BasicBlockProto containing the instructions from `machine_code`.

          Raises:
            StatusNotOk: When extracting instructions from the machine code
              fails or `machine_code` does not have the right format.)"
          // TODO(ondrasej): Raise ValueError when `machine_code` does not have
          // the right format.
          )
      .def(  //
          "basic_block_with_throughput_proto_from_csv_line",
          &BHiveImporter::ParseBHiveCsvLine, py::arg("source_name"),
          py::arg("line"), py::arg("machine_code_hex_column_index") = 0,
          py::arg("throughput_column_index") = 1,
          py::arg("throughput_scaling") = 1.0,
          py::arg("base_address") = uint64_t{0},
          R"(Creates a BasicBlockWithThroughputProto from a BHive CSV line.

          Takes a string in the format "{machine_code},{throughput}" where
          {machine_code} is the machine code of a basic block in the format
          accepted by `basic_block_proto_from_hex`, and {throughput} is the
          measured throughput of the basic block.

          Args:
            source_name: The name of the throughput source used in the output
              proto.
            line: The line from the BHive CSV file.
            throughput_scaling: An optional scaling applied to {throughput}.
            base_address: The address of the first instruction of the basic
              block.

          Returns:
            A BasicBlockWithThroughputProto that contains the basic block
            extracted from {machine_code}, and one throughput information based
            on {throughput}.

          Raises:
            StatusNotOk: When parsing the CSV line or extracting data from the
              machine code fails.)");
}

}  // namespace gematria
