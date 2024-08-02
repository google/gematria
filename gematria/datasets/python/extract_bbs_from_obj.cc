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

#include <memory>
#include <string_view>

#include "gematria/datasets/extract_bbs_from_obj_lib.h"
#include "gematria/llvm/llvm_to_absl.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"  // IWYU pragma: keep
#include "pybind11_abseil/import_status_module.h"
#include "pybind11_abseil/status_casters.h"  // IWYU pragma: keep

namespace gematria {

namespace py = ::pybind11;

PYBIND11_MODULE(extract_bbs_from_obj, m) {
  m.doc() = "Code for extracting BBs from object files and binaries.";

  py::google::ImportStatusModule();

  m.def(
      "get_basic_block_hex_values",
      [](py::bytes binary_data) {
        // We need to convert the bytes object into a MemoryBufferRef that we
        // can pass along to getBasicBlockHexValue
        std::string_view binary_data_view = binary_data;
        StringRef binary_data_ref(binary_data_view.data(),
                                  binary_data_view.size());
        std::unique_ptr<llvm::MemoryBuffer> memory_buffer =
            llvm::MemoryBuffer::getMemBuffer(binary_data_ref);

        return LlvmExpectedToStatusOr(getBasicBlockHexValues(*memory_buffer));
      },
      py::arg("binary_data"),
      R"(Returns an array of basic block hex values from a binary.

      Processes binary data in the form of an ELF executable or object file
      and returns a list of basic blocks contained within the binary in
      hex format. Assumes that the binary has been compiled with basic block
      address maps enabled.

      Args:
        binary_data: A `bytes` object that contains the binary/object file.
      
      Returns:
        A list of strings containing the basic block hex values.
      
      Raises:
        StatusNotOk: When processing the binary fails.
      )");
}

}  // namespace gematria
