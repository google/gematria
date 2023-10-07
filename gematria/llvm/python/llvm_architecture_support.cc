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

#include "gematria/llvm/llvm_architecture_support.h"

#include "gematria/llvm/llvm_to_absl.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
#include "pybind11_abseil/import_status_module.h"
#include "pybind11_abseil/status_casters.h"

namespace gematria {

namespace py = ::pybind11;

PYBIND11_MODULE(llvm_architecture_support, m) {
  m.doc() = "Provides helper class for passing around LLVM target objects.";

  py::google::ImportStatusModule();

  py::class_<LlvmArchitectureSupport>(
      m, "LlvmArchitectureSupport",
      R"(Opaque holder class for LLVM target objects.

      This class does not have any public methods or attributes in Python. It is
      only meant to be used as a token passed to Gematria C++ classes and
      functions that use LLVM APIs and expect LLVM objects as their inputs.)")
      .def_static(  //
          "from_triple",
          [](std::string_view llvm_triple, std::string_view cpu,
             std::string_view cpu_features) {
            return LlvmExpectedToStatusOr(LlvmArchitectureSupport::FromTriple(
                llvm_triple, cpu, cpu_features));
          },
          py::arg("llvm_triple"), py::arg("cpu") = std::string(),
          py::arg("cpu_features") = std::string(),
          R"(Returns a new LlvmArchitectureSupport from an LLVM triple.

          Args:
            llvm_triple: The LLVM triple for which the object should be created.
            cpu: Optional. The name of the CPU for which the object should be
              created.
            cpu_features: Optional. An LLVM CPU feature string further
              specifying the parameters of the created object.

          Raises:
            StatusNotOk: When the LLVM triple does not correspond to a LLVM
              target architecture supported by Gematria.)")
      .def_static("x86_64", &LlvmArchitectureSupport::X86_64,
                  R"(Returns a new LlvmArchitectureSupport for x86-64.)");
}

}  // namespace gematria
