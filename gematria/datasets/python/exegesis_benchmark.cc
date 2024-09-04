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

#include <memory>

#include "absl/status/statusor.h"
#include "gematria/datasets/bhive_to_exegesis.h"
#include "gematria/datasets/exegesis_benchmark_lib.h"
#include "gematria/llvm/llvm_to_absl.h"
#include "gematria/proto/execution_annotation.pb.h"
#include "llvm-c/Target.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkCode.h"
#include "llvm/tools/llvm-exegesis/lib/PerfHelper.h"
#include "llvm/tools/llvm-exegesis/lib/TargetSelect.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11_abseil/import_status_module.h"
#include "pybind11_abseil/status_casters.h"         // IWYU pragma: keep
#include "pybind11_protobuf/native_proto_caster.h"  // IWYU pragma: keep

namespace gematria {
namespace {

void InitializeForExegesisOnce() {
  static bool initialize_internals = []() {
    // LLVM Setup
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86Target();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86AsmParser();
    LLVMInitializeX86Disassembler();

    // Exegesis Setup
    InitializeX86ExegesisTarget();

    if (pfm::pfmInitialize()) return false;

    return true;
  }();
  (void)initialize_internals;
}

}  // namespace

namespace py = ::pybind11;

using namespace llvm;
using namespace llvm::exegesis;

PYBIND11_MODULE(exegesis_benchmark, m) {
  m.doc() = "Code for benchmarking basic blocks.";

  py::google::ImportStatusModule();

  py::class_<BenchmarkCode>(m, "BenchmarkCode");

  py::class_<ExegesisBenchmark>(m, "ExegesisBenchmark")
      .def("create",
           []() -> absl::StatusOr<std::unique_ptr<ExegesisBenchmark>> {
             InitializeForExegesisOnce();

             return LlvmExpectedToStatusOr(ExegesisBenchmark::create());
           })
      .def("process_annotated_block",
           [](ExegesisBenchmark& Self,
              const BlockWithExecutionAnnotations& Annotations) {
             return LlvmExpectedToStatusOr(Self.processAnnotatedBlock(
                 Annotations.block_hex(), Annotations.execution_annotations()));
           })
      .def("benchmark_basic_block", [](ExegesisBenchmark& Self,
                                       const BenchmarkCode& InputBenchmark) {
        return LlvmExpectedToStatusOr(Self.benchmarkBasicBlock(InputBenchmark));
      });
}

}  // namespace gematria
