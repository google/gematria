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

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gematria/datasets/bhive_importer.h"
#include "gematria/datasets/find_accessed_addrs_exegesis.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/proto/basic_block.pb.h"
#include "gematria/proto/execution_annotation.pb.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Support/Error.h"
#include "llvm/tools/llvm-exegesis/lib/LlvmState.h"

namespace gematria {

struct AnnotatedBlock {
  ExecutionAnnotations AccessedAddrs;
  BasicBlockProto BasicBlockProto;
};

class BHiveToExegesis {
 public:
  enum class AnnotatorType { kExegesis, kFast, kNone };

  static constexpr std::pair<AnnotatorType, std::string_view>
      kAnnotatorTypeNames[] = {{AnnotatorType::kExegesis, "exegesis"},
                               {AnnotatorType::kFast, "fast"},
                               {AnnotatorType::kNone, "none"}};

 private:
  std::unique_ptr<ExegesisAnnotator> LLVMAnnotator;
  llvm::exegesis::LLVMState ExegesisState;
  gematria::X86Canonicalizer Canonicalizer;
  gematria::BHiveImporter BHiveImporter;
  LlvmArchitectureSupport &ArchSupport;
  std::unique_ptr<llvm::MCInstPrinter> InstPrinter;

  BHiveToExegesis(
      LlvmArchitectureSupport &ArchitectureSupport,
      llvm::exegesis::LLVMState &&LLVMExegesisState,
      std::unique_ptr<gematria::ExegesisAnnotator> &&LLVMExegesisAnnotator);

  absl::StatusOr<ExecutionAnnotations> getAccessedAddrs(
      absl::Span<const uint8_t> BasicBlock,
      const unsigned MaxAnnotationAttempts, AnnotatorType AnnotatorToUse);

 public:
  static llvm::Expected<std::unique_ptr<BHiveToExegesis>> create(
      LlvmArchitectureSupport &ArchitectureSupport);
  absl::StatusOr<AnnotatedBlock> annotateBasicBlock(
      std::string_view BasicBlockHex, AnnotatorType AnnotatorToUse,
      const unsigned MaxAnnotationAttempts);
};

bool AbslParseFlag(absl::string_view text, BHiveToExegesis::AnnotatorType *type,
                   std::string *error);
std::string AbslUnparseFlag(BHiveToExegesis::AnnotatorType type);

}  // namespace gematria
