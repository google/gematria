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
#include "pybind11/cast.h"
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
      .def(
          "create",
          []() -> absl::StatusOr<std::unique_ptr<ExegesisBenchmark>> {
            InitializeForExegesisOnce();

            return LlvmExpectedToStatusOr(ExegesisBenchmark::create());
          },
          R"(Creates an ExegesisBenchmark Instance.

          Does the necessary initialization to run Exegesis and creates an
          ExegesisBenchmark that can be used to execute annotated blocks. This
          will only perform initialization once even if called multiple times,
          but it is reccomended to only create one instance per thread/worker.

          Returns:
            An ExegesisBenchmark Instance.
           
          Raises:
            StatusNotOk: When creating the ExegesisBenchmark instance fails.
          )")
      .def(
          "process_annotated_block",
          [](ExegesisBenchmark& Self,
             const BlockWithExecutionAnnotations& BlockAndAnnotations) {
            return LlvmExpectedToStatusOr(Self.processAnnotatedBlock(
                BlockAndAnnotations.block_hex(),
                BlockAndAnnotations.execution_annotations()));
          },
          py::arg("block_and_annotations"),
          R"(Processes an annotated block into an executable form.

          Takes a BlockWithExecutionAnnotations proto and converts it into a
          BenchmarkCode instance, which can be understood by Exegesis and thus
          executed/benchmarked.

          Args:
            block_and_annotations: A BlockWithExecutionAnnotations proto
              containing the block of interest and associated annotations.
          
          Returns:
            A BenchmarkCode instance.
          
          Raises:
            StatusNotOk: When converting the block to a BenchmarkCode instance
              fails.
          )")
      .def(
          "benchmark_basic_block",
          [](ExegesisBenchmark& Self, const BenchmarkCode& InputBenchmark) {
            return LlvmExpectedToStatusOr(
                Self.benchmarkBasicBlock(InputBenchmark));
          },
          py::arg("input_benchmark"),
          R"(Benchmarks a block in the form of a BenchmarkCode instance.

          Takes a BenchmarkCode instance and then executes it, collecting
          performance information at the same time.

          Args:
            input_benchmark: The BenchmarkCode instance formed from the block and
              annotations of interest that should be benchmarked.
          
          Returns:
            A floating point value representing the inverse throughput (i.e.,
            the number of cycles needed to execute the block in steady state)
            of the block.
          
          Raises:
            StatusNotOk: When benchmarking the block fails.
          )");
}

}  // namespace gematria
