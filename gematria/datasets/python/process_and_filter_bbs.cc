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

#include <string>

#include "gematria/datasets/process_and_filter_bbs_lib.h"
#include "gematria/llvm/llvm_to_absl.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11_abseil/import_status_module.h"
#include "pybind11_abseil/status_casters.h"  // IWYU pragma: keep

namespace gematria {

namespace py = ::pybind11;

PYBIND11_MODULE(process_and_filter_bbs, m) {
  m.doc() =
      "Code for processing and filtering BBs during dataset construction.";

  py::google::ImportStatusModule();

  py::class_<BBProcessorFilter>(m, "BBProcessorFilter")
      .def(py::init<>(),
           R"(Initializes a new BBProcessorFilter.
    )")
      .def(
          "remove_risky_instructions",
          [](BBProcessorFilter& Self, std::string BasicBlock,
             std::string Filename, bool FilterMemoryAccessingBlocks) {
            return LlvmExpectedToStatusOr(Self.removeRiskyInstructions(
                BasicBlock, Filename, FilterMemoryAccessingBlocks));
          },
          py::arg("basic_block"), py::arg("file_name"),
          py::arg("filter_memory_accessing_blocks"),
          R"(Processes a raw basic block extracted from an ELF file.

          Processes and filters a raw basic block, performing operations such
          as removing calls and branches so that the BB can then be annotated
          and benchmarked without violating any modeling assumptions:

          Args:
            basic_block: A string containing the hex representation of a basic
              block.
            file_name: The name of the file that the basic block came from, or
              some other identifier.
            filter_memory_accessing_blocks: Whether or not to filter memory
              accessing instructions like loads and stores.
          
          Returns:
            A hex string containing the basic block that has been processed.
          
          Raises:
            StatusNotOk: When processing the basic block fails.
          )");
}

}  // namespace gematria
