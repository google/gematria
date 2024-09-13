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

#include "absl/status/statusor.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/llvm/llvm_to_absl.h"
#include "gematria/proto/execution_annotation.pb.h"
#include "llvm-c/Target.h"
#include "llvm/tools/llvm-exegesis/lib/TargetSelect.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"  // IWYU pragma: keep
#include "pybind11_abseil/import_status_module.h"
#include "pybind11_abseil/status_casters.h"         // IWYU pragma: keep
#include "pybind11_protobuf/native_proto_caster.h"  // IWYU pragma: keep

namespace gematria {

namespace py = ::pybind11;

PYBIND11_MODULE(bhive_to_exegesis, m) {
  m.doc() = "Code for annotating basic blocks.";

  py::google::ImportStatusModule();

  py::enum_<BHiveToExegesis::AnnotatorType>(m, "AnnotatorType")
      .value("exegesis", BHiveToExegesis::AnnotatorType::kExegesis)
      .value("fast", BHiveToExegesis::AnnotatorType::kFast)
      .value("none", BHiveToExegesis::AnnotatorType::kNone)
      .export_values();

  py::class_<BHiveToExegesis>(m, "BHiveToExegesis")
      .def_static(
          "create",
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
          },
          py::arg("architecture_support"),
          R"(Creates a BHiveToExegesis Instance.

          Performs the relevant LLVM setup and creates a BHiveToExegesis
          instance that can then be used to annotate basic blocks.

          Args:
            architecture_support: An LLVMArchitectureSupport instance
              containing the relevant helper classes.
           
          Returns:
            A BHiveToExegsis Instance.
          
          Raises:
            StatusNotOk: When creating the BHiveToExegesis instance fails.
          )")
      .def(
          "annotate_basic_block",
          [](BHiveToExegesis& Self, std::string BasicBlockHex,
             BHiveToExegesis::AnnotatorType AnnotatorType,
             unsigned MaxAnnotationAttempts)
              -> absl::StatusOr<ExecutionAnnotations> {
            absl::StatusOr<AnnotatedBlock> annotated_block =
                Self.annotateBasicBlock(BasicBlockHex, AnnotatorType,
                                        MaxAnnotationAttempts);
            if (!annotated_block.ok()) {
              return annotated_block.status();
            }

            return annotated_block->AccessedAddrs;
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
            An ExecutionAnnotations proto that can be used for benchmarking.

          Raises:
            StatusNotOk: When annotating the block fails.
          )");
}

}  // namespace gematria
