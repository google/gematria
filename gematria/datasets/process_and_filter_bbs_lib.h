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

#include <string>

#include "gematria/llvm/llvm_architecture_support.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Support/Error.h"

namespace gematria {

// BBProcessorFilter is used for processing and filtering raw basic blocks
// extracted directly from applications. It contains functions to process
// individual blocks with the class containing the relevant stateful objects
// needed for processing.
class BBProcessorFilter {
 public:
  BBProcessorFilter();

  // Removes instructions that might be considered risky in that they
  // might violate modeling assumptions explicitly or have unmodeled
  // side effects that might cause problems
  llvm::Expected<std::string> removeRiskyInstructions(
      const llvm::StringRef BasicBlock, const llvm::StringRef Filename,
      bool FilterMemoryAccessingBlocks);

 private:
  std::unique_ptr<LlvmArchitectureSupport> LLVMSupport;
  std::unique_ptr<llvm::MCInstPrinter> InstructionPrinter;
};

}  // namespace gematria
