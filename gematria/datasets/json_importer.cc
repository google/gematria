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

#include "gematria/datasets/json_importer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

#include "absl/status/statusor.h"
#include "gematria/datasets/bhive_importer.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/llvm_to_absl.h"
#include "gematria/proto/annotation.pb.h"
#include "gematria/proto/basic_block.pb.h"
#include "gematria/proto/throughput.pb.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

namespace gematria {

JSONImporter::JSONImporter(const Canonicalizer* canonicalizer)
    : bhive_importer_(std::make_unique<BHiveImporter>(canonicalizer)) {}

absl::StatusOr<BasicBlockWithThroughputProto> JSONImporter::ParseJSON(
    std::string_view source_name, std::string_view json_string,
    double throughput_scaling /* = 1.0 */, uint64_t base_address /* = 0 */) {
  llvm::Expected<llvm::json::Value> json_value =
      llvm::json::parse(llvm::StringRef(json_string));
  if (llvm::Error error = json_value.takeError()) {
    return LlvmErrorToStatus(std::move(error));
  }
  llvm::json::Object* json_block = json_value->getAsObject();

  std::optional<llvm::StringRef> machine_code_hex =
      json_block->getString("machine_code_hex");
  if (!machine_code_hex.has_value()) {
    return absl::InvalidArgumentError(
        "Basic block JSON is missing key `machine_code_hex`.");
  }
  BasicBlockWithThroughputProto proto;
  absl::StatusOr<BasicBlockProto> block_proto_or_status =
      bhive_importer_->BasicBlockProtoFromMachineCodeHex(*machine_code_hex,
                                                         base_address);
  if (!block_proto_or_status.ok()) return block_proto_or_status.status();
  *proto.mutable_basic_block() = std::move(block_proto_or_status).value();

  llvm::json::Array* json_annotations =
      json_block->getArray("instruction_annotations");
  auto instructions =
      proto.mutable_basic_block()->mutable_canonicalized_instructions();
  for (llvm::json::Value raw_json_annotation : *json_annotations) {
    llvm::json::Object* json_annotation = raw_json_annotation.getAsObject();
    std::optional<llvm::StringRef> annotation_name =
        json_annotation->getString("name");
    if (!annotation_name.has_value()) {
      return absl::InvalidArgumentError(
          "Basic block annotation JSON is missing key `name`.");
    }
    llvm::json::Array* json_annotation_values =
        json_annotation->getArray("values");
    if (json_annotation_values->size() !=
        proto.basic_block().canonicalized_instructions_size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Basic block annotation JSON does not have the same number of "
          "annotation values as instructions for annotation name: ",
          (*annotation_name).data()));
    }
    for (int i = 0; i < instructions->size(); ++i) {
      std::optional<double> annotation_value =
          (*json_annotation_values)[i].getAsNumber();
      if (!annotation_value.has_value()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Basic block annotation value JSON cannot be "
            "intepreted as double for annotation name: ",
            (*annotation_name).data(), " and instruction index: ", i));
      }
      AnnotationProto* annotation_proto =
          (*instructions)[i].add_instruction_annotations();
      annotation_proto->set_name(*annotation_name);
      annotation_proto->set_value(*annotation_value);
    }
  }

  std::optional<double> throughput_cycles = json_block->getNumber("throughput");
  if (!throughput_cycles.has_value()) {
    return absl::InvalidArgumentError(
        "Basic block JSON is missing key `throughput`.");
  }

  ThroughputWithSourceProto& throughput = *proto.add_inverse_throughputs();
  throughput.set_source(source_name);
  throughput.add_inverse_throughput_cycles(*throughput_cycles *
                                           throughput_scaling);

  return proto;
}

}  // namespace gematria