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

// Use the absolute path for headers from llvm-exegesis as there is no
// canonical include path within LLVM as they are not properly exposed through
// a library and could potentially be confused with other LLVM includes.

#include "gematria/datasets/find_accessed_addrs.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkRunner.h"
#include "llvm/tools/llvm-exegesis/lib/LlvmState.h"

using namespace llvm;
using namespace llvm::exegesis;

namespace gematria {

class ExegesisAnnotator {
  std::unique_ptr<MCInstPrinter> MachinePrinter;
  std::unique_ptr<MCContext> MachineContext;
  std::unique_ptr<MCDisassembler> MachineDisassembler;

  LLVMState &State;
  std::unique_ptr<BenchmarkRunner> Runner;
  std::unique_ptr<const SnippetRepetitor> Repetitor;

  ExegesisAnnotator(LLVMState &ExegesisState,
                    std::unique_ptr<BenchmarkRunner> BenchRunner,
                    std::unique_ptr<const SnippetRepetitor> SnipRepetitor);

 public:
  static Expected<std::unique_ptr<ExegesisAnnotator>> create(
      LLVMState &ExegesisState);
  Expected<BlockAnnotations> findAccessedAddrs(ArrayRef<uint8_t> BasicBlock,
                                               unsigned MaxAnnotationAttempts);
};

}  // namespace gematria

#endif  // GEMATRIA_DATASETS_FIND_ACCESSED_ADDRS_EXEGESIS_H_
