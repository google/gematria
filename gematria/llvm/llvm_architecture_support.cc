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

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/include/llvm-c/Target.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "llvm/include/llvm/MC/MCContext.h"
#include "llvm/include/llvm/MC/TargetRegistry.h"
#include "llvm/include/llvm/Target/TargetOptions.h"

namespace gematria {
namespace {

void InitializeLlvmOnce() {
  static bool initialize_llvm_internals = []() {
    // Initialize LLVM for all supported architectures. As of 2023-04, only
    // x86-64 is supported.
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86AsmParser();
    LLVMInitializeX86Disassembler();
    return true;
  }();
  CHECK(initialize_llvm_internals);
}

}  // namespace

absl::StatusOr<std::unique_ptr<LlvmArchitectureSupport>>
LlvmArchitectureSupport::FromTriple(std::string_view llvm_triple,
                                    std::string_view cpu,
                                    std::string_view cpu_features) {
  // This point is on the critical path for all uses of LlvmArchitectureSupport.
  // Initialize LLVM for all supported architectures on the first call.
  InitializeLlvmOnce();
  std::string lookup_error;
  // TODO(ondrasej): Remove the std::string() conversion once it's no longer
  // needed.
  const llvm::Target* const llvm_target = llvm::TargetRegistry::lookupTarget(
      std::string(llvm_triple), lookup_error);
  if (llvm_target == nullptr) {
    return absl::NotFoundError(
        absl::StrCat("Could not find target for triple: ", llvm_triple));
  }

  return std::unique_ptr<LlvmArchitectureSupport>(
      new LlvmArchitectureSupport(llvm_triple, cpu, cpu_features, llvm_target));
}

std::unique_ptr<LlvmArchitectureSupport> LlvmArchitectureSupport::X86_64() {
  absl::StatusOr<std::unique_ptr<LlvmArchitectureSupport>> x86_64_or_status =
      FromTriple("x86_64", "", "");
  CHECK_OK(x86_64_or_status);
  return std::move(x86_64_or_status).value();
}

LlvmArchitectureSupport::LlvmArchitectureSupport(std::string_view llvm_triple,
                                                 std::string_view cpu,
                                                 std::string_view cpu_features,
                                                 const llvm::Target* target)
    : target_(ABSL_DIE_IF_NULL(target)) {
  llvm::TargetOptions target_options;
  target_machine_.reset(target_->createTargetMachine(
      /*TT=*/llvm_triple, /*CPU=*/cpu, /*Features=*/cpu_features,
      /*Options=*/target_options, /*RM=*/std::nullopt));

  mc_context_ = std::make_unique<llvm::MCContext>(
      target_machine_->getTargetTriple(), target_machine_->getMCAsmInfo(),
      target_machine_->getMCRegisterInfo(),
      target_machine_->getMCSubtargetInfo());
  mc_disassembler_.reset(target_->createMCDisassembler(
      *target_machine_->getMCSubtargetInfo(), *mc_context_));
}

}  // namespace gematria
