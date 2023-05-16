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

#include "gematria/llvm/canonicalizer.h"

#include <memory>

#include "gematria/llvm/llvm_architecture_support.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"

namespace gematria {

namespace py = ::pybind11;

PYBIND11_MODULE(canonicalizer, m) {
  m.doc() =
      "Provides code for extracting canonical representation of instructions";

  py::class_<Canonicalizer>(  //
      m, "Canonicalizer",
      R"(Provides methods for extracting canonicalized instructions.

      Does not export any methods in Python; this class is meant to be passed to
      functions and classes imported from C++ that need a canonicalizer.)")
      .def_static(
          "x86_64",
          [](const LlvmArchitectureSupport& llvm_architecture) {
            // We need to explicitly convert to std::unique_ptr<Canonicalizer>
            // to make sure that pybind11 knows how to match the types.
            return std::unique_ptr<Canonicalizer>(
                std::make_unique<X86Canonicalizer>(
                    &llvm_architecture.target_machine()));
          },
          py::arg("llvm_architecture"),
          "Creates a new Canonicalizer for x86-64 from the given LLVM "
          "architecture support");
}

}  // namespace gematria
