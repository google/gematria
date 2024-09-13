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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gematria/datasets/bhive_importer.h"
#include "gematria/datasets/find_accessed_addrs.h"
#include "gematria/datasets/find_accessed_addrs_exegesis.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/llvm/llvm_to_absl.h"
#include "gematria/proto/execution_annotation.pb.h"
#include "gematria/utils/string.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/tools/llvm-exegesis/lib/LlvmState.h"

using namespace llvm::exegesis;
using namespace llvm;

namespace gematria {

BHiveToExegesis::BHiveToExegesis(
    LlvmArchitectureSupport& ArchitectureSupport,
    llvm::exegesis::LLVMState&& LLVMExegesisState,
    std::unique_ptr<ExegesisAnnotator>&& LLVMExegesisAnnotator)
    : LLVMAnnotator(std::move(LLVMExegesisAnnotator)),
      ExegesisState(std::move(LLVMExegesisState)),
      Canonicalizer(&ArchitectureSupport.target_machine()),
      BHiveImporter(&Canonicalizer),
      ArchSupport(ArchitectureSupport),
      InstPrinter(ArchSupport.CreateMCInstPrinter(0)) {}

Expected<std::unique_ptr<BHiveToExegesis>> BHiveToExegesis::create(
    LlvmArchitectureSupport& ArchitectureSupport) {
  Expected<LLVMState> LLVMStateOrErr = LLVMState::Create("", "native");
  if (!LLVMStateOrErr) return LLVMStateOrErr.takeError();

  Expected<std::unique_ptr<ExegesisAnnotator>> AnnotatorOrErr =
      ExegesisAnnotator::create(*LLVMStateOrErr);
  if (!AnnotatorOrErr) return AnnotatorOrErr.takeError();

  return std::unique_ptr<BHiveToExegesis>(
      new BHiveToExegesis(ArchitectureSupport, std::move(*LLVMStateOrErr),
                          std::move(*AnnotatorOrErr)));
}

absl::StatusOr<ExecutionAnnotations> BHiveToExegesis::getAccessedAddrs(
    absl::Span<const uint8_t> BasicBlock, const unsigned MaxAnnotationAttempts,
    AnnotatorType AnnotatorToUse) {
  switch (AnnotatorToUse) {
    case AnnotatorType::kFast:
      return gematria::FindAccessedAddrs(BasicBlock, ArchSupport);
    case AnnotatorType::kExegesis:
      return LlvmExpectedToStatusOr(LLVMAnnotator->findAccessedAddrs(
          ArrayRef(BasicBlock.begin(), BasicBlock.end()),
          MaxAnnotationAttempts));
    case AnnotatorType::kNone:
      return gematria::ExecutionAnnotations();
  }
  return absl::InvalidArgumentError("unknown annotator type");
}

absl::StatusOr<AnnotatedBlock> BHiveToExegesis::annotateBasicBlock(
    std::string_view BasicBlockHex, AnnotatorType AnnotatorToUse,
    const unsigned MaxAnnotationAttempts) {
  std::optional<std::vector<uint8_t>> Bytes =
      gematria::ParseHexString(BasicBlockHex);
  if (!Bytes.has_value())
    return absl::InvalidArgumentError(
        Twine("Could not parse ").concat(BasicBlockHex).str());

  llvm::Expected<std::vector<gematria::DisassembledInstruction>> Instructions =
      gematria::DisassembleAllInstructions(
          ArchSupport.mc_disassembler(), ArchSupport.mc_instr_info(),
          ArchSupport.mc_register_info(), ArchSupport.mc_subtarget_info(),
          *InstPrinter, 0, *Bytes);

  if (!Instructions) {
    return absl::InvalidArgumentError(
        Twine("Failed to disassemble block ").concat(BasicBlockHex).str());
  }

  auto Proto = BHiveImporter.BasicBlockProtoFromInstructions(*Instructions);

  auto Annotations =
      getAccessedAddrs(*Bytes, MaxAnnotationAttempts, AnnotatorToUse);

  if (!Annotations.ok()) return Annotations.status();

  AnnotatedBlock annotated_block;
  annotated_block.AccessedAddrs = std::move(*Annotations);
  annotated_block.BasicBlockProto = std::move(Proto);

  return std::move(annotated_block);
}

bool AbslParseFlag(absl::string_view text, BHiveToExegesis::AnnotatorType* type,
                   std::string* error) {
  for (const auto& [annotator_type, type_string] :
       BHiveToExegesis::kAnnotatorTypeNames) {
    if (text == type_string) {
      *type = annotator_type;
      return true;
    }
  }

  *error = "unknown annotator type";
  return false;
}

std::string AbslUnparseFlag(BHiveToExegesis::AnnotatorType type) {
  for (const auto& [annotator_type, type_string] :
       BHiveToExegesis::kAnnotatorTypeNames) {
    if (annotator_type == type) return std::string(type_string);
  }

  __builtin_unreachable();
}

}  // namespace gematria
