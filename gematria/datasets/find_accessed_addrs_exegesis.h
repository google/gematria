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

#ifndef GEMATRIA_DATASETS_FIND_ACCESSED_ADDRS_EXEGESIS_H_
#define GEMATRIA_DATASETS_FIND_ACCESSED_ADDRS_EXEGESIS_H_

#include <cstdint>
#include <vector>

#include "BenchmarkRunner.h"
#include "LlvmState.h"
#include "gematria/datasets/find_accessed_addrs.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "llvm/ADT/ArrayRef.h"

namespace gematria {

class ExegesisAnnotator {
  LlvmArchitectureSupport &ArchSupport;
  std::unique_ptr<llvm::MCInstPrinter> MCPrinter;

  llvm::exegesis::LLVMState &State;
  std::unique_ptr<llvm::exegesis::BenchmarkRunner> Runner;
  std::unique_ptr<const llvm::exegesis::SnippetRepetitor> Repetitor;

  ExegesisAnnotator(
      LlvmArchitectureSupport &ArchSupport_, llvm::exegesis::LLVMState &State_,
      std::unique_ptr<llvm::exegesis::BenchmarkRunner> Runner_,
      std::unique_ptr<const llvm::exegesis::SnippetRepetitor> Repetitor_);

 public:
  static llvm::Expected<std::unique_ptr<ExegesisAnnotator>> Create(
      LlvmArchitectureSupport &ArchSupport_, llvm::exegesis::LLVMState &State_);
  llvm::Expected<AccessedAddrs> FindAccessedAddrs(
      llvm::ArrayRef<uint8_t> BasicBlock);
};

}  // namespace gematria

#endif  // GEMATRIA_DATASETS_FIND_ACCESSED_ADDRS_EXEGESIS_H_
