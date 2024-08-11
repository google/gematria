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

#include "gematria/datasets/bhive_to_exegesis.h"

#include <string>

#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/llvm/llvm_to_absl.h"
#include "llvm-c/Target.h"
#include "llvm/tools/llvm-exegesis/lib/TargetSelect.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"  // IWYU pragma: keep
#include "pybind11_abseil/import_status_module.h"
#include "pybind11_abseil/status_casters.h"  // IWYU pragma: keep

namespace gematria {

namespace py = ::pybind11;

PYBIND11_MODULE(bhive_to_exegesis, m) {
  m.doc() = "Code for annotating baisc blocks.";

  py::google::ImportStatusModule();

  py::enum_<BHiveToExegesis::AnnotatorType>(m, "AnnotatorType")
      .value("exegesis", BHiveToExegesis::AnnotatorType::kExegesis)
      .value("fast", BHiveToExegesis::AnnotatorType::kFast)
      .value("none", BHiveToExegesis::AnnotatorType::kNone)
      .export_values();

  py::class_<AnnotatedBlock>(m, "AnnotatedBlock");

  py::class_<BHiveToExegesis>(m, "BHiveToExegesis")
      .def("create",
           [](LlvmArchitectureSupport& ArchitectureSupport) {
             LLVMInitializeX86Target();
             LLVMInitializeX86TargetInfo();
             LLVMInitializeX86TargetMC();
             LLVMInitializeX86AsmPrinter();
             LLVMInitializeX86AsmParser();
             LLVMInitializeX86Disassembler();
             InitializeX86ExegesisTarget();

             return LlvmExpectedToStatusOr(
                 BHiveToExegesis::create(ArchitectureSupport));
           })
      .def(
          "annotate_basic_block",
          [](BHiveToExegesis& Self, std::string BasicBlockHex,
             BHiveToExegesis::AnnotatorType AnnotatorType,
             unsigned MaxAnnotationAttempts) {
            return Self.annotateBasicBlock(BasicBlockHex, AnnotatorType,
                                           MaxAnnotationAttempts);
          },
          py::arg("basic_block_hex"), py::arg("annotator_type"),
          py::arg("max_annotation_attempts"),
          R"(Annotates a basic block, providing memory and register info.

          Takes a single basic block and annotates it, providing information
          on initial register values and the memory setup that should be used
          in order to execute the block.

          Args:
            basic_block_hex: The basic block in hexadecimal format.
            annotator_type: The annotator to use.
            max_annotation_attempts: The maximum number of times to try
              to find an appropriate memory setup for the block before
              giving up.
            
          Returns:
            An AnnotatedBlock that can then be used for benchmarking.

          Raises:
            StatusNotOk: When annotating the block fails.
          )");
}

}  // namespace gematria
