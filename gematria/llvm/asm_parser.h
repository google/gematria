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

#ifndef THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_ASM_PARSER_H_
#define THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_ASM_PARSER_H_

#include <memory>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"

namespace gematria {

// Parses `assembly` from a string to a sequence of MCInst. Uses the given ASM
// dialect for parsing. Labels, directives, and other parts of the assembly
// language that are not instructions are ignored by the function.
absl::StatusOr<std::vector<llvm::MCInst>> ParseAsmCodeFromString(
    const llvm::TargetMachine& target_machine, std::string_view assembly,
    llvm::InlineAsm::AsmDialect dialect);

// A version of ParseAsmCodeFromBuffer that takes an llvm::MemoryBuffer instead
// of a string.
absl::StatusOr<std::vector<llvm::MCInst>> ParseAsmCodeFromBuffer(
    const llvm::TargetMachine& target_machine,
    std::unique_ptr<llvm::MemoryBuffer> buffer,
    llvm::InlineAsm::AsmDialect dialect);

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_ASM_PARSER_H_
