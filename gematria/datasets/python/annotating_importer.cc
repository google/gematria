// Copyright 2024 Google Inc.
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

#include "gematria/datasets/annotating_importer.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "gematria/llvm/canonicalizer.h"
#include "llvm/Support/Error.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11_abseil/import_status_module.h"
#include "pybind11_abseil/status_casters.h"
#include "pybind11_protobuf/native_proto_caster.h"

namespace gematria {

namespace py = ::pybind11;

PYBIND11_MODULE(annotating_importer, m) {
  m.doc() = "Support code for importing annotated basic block data.";

  py::google::ImportStatusModule();

  py::class_<AnnotatingImporter>(m, "AnnotatingImporter")
      .def(  //
          py::init<const Canonicalizer* /* canonicalizer */>(),
          py::arg("canonicalizer"), py::keep_alive<1, 2>(),
          R"(Initializes a new annotation collector for a given architecture.
           
          Args:
            canonicalizer: The canonicalizer used to disassemble instructions
              and convert them to the Gematria proto representation.)")
      .def(  //
          "get_annotated_basic_block_protos",
          &AnnotatingImporter::GetAnnotatedBasicBlockProtos,
          py::arg("elf_file_name"), py::arg("perf_data_file_name"),
          py::arg("source_name"),
          R"(Creates annotated BasicBlockProtos from an ELF object and samples.
          
          Reads an ELF object along with a corresponding `perf.data`-like file
          and creates a list of annotated `BasicBlockProto`s consisting of
          basic blocks from the ELF object annotated using samples from the
          `perf.data`-like file.
          
          Args:
            elf_file_name: The path to the ELF object from which basic blocks
              are to be extracted.
            perf_data_file_name: The path to the `perf.data`-like file from
              which samples are to be extracted along with LBR data.
            source_name: The source name the timing data in the annotated
              `BasicBlockProto`s should be attributed to.
              
          Returns:
            A list of annotated `BasicBlockProto`s.
            
          Raises:
            StatusNotOk: When extracting basic blocks and samples or creating
              the annotated `BasicBlockProto`s fails.)");
}

}  // namespace gematria
