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

#ifndef GEMATRIA_DATASETS_JSON_IMPORTER_H_
#define GEMATRIA_DATASETS_JSON_IMPORTER_H_

#include <cstdint>
#include <memory>
#include <string_view>

#include "absl/status/statusor.h"
#include "gematria/datasets/bhive_importer.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/proto/throughput.pb.h"

namespace gematria {

// Parser for JSON objects representing basic blocks.
class JSONImporter {
 public:
  // Creates a new BHive importer from a given canonicalizer. The canonicalizer
  // must be for the architecture/microarchitecture of the data set.
  // Does not take ownership of the canonicalizer.
  explicit JSONImporter(const Canonicalizer* canonicalizer);

  // Parses JSON objects representing basic blocks, Expects that the JSON
  // follows the format:
  // {
  //   "machine_code_hex": <machine_code_hex>,
  //   "instruction_annotations": [
  //     { "name": <annotation_type_name>,
  //       "values": [<annotation_value>, ...] },
  //     ...
  //   ],
  //   "throughput": <throughput>
  // }
  // where <machine_code_hex> is a hex string holding the machine code that the
  // basic block is comprised of, <annotation_type_name> is a string holding the
  // name of the type of annotation represented by this element of
  // "instruction_annotations", <annotation_value> is a numeric value
  // corresponding to the annotation type for the instruction sharing the same
  // index, and <throughput> is the numeric inverse throughput of the basic
  // block. Optionally applies `throughput_scaling` to the throughput value, and
  // uses `base_address` as the address of the first instruction in the basic
  // block.
  absl::StatusOr<BasicBlockWithThroughputProto> ParseJSON(
      std::string_view source_name, std::string_view json_string,
      double throughput_scaling = 1.0, uint64_t base_address = 0);

 private:
  std::unique_ptr<BHiveImporter> bhive_importer_;
};

}  // namespace gematria

#endif  // GEMATRIA_DATASETS_JSON_IMPORTER_H_