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

#ifndef THIRD_PARTY_GEMATRIA_GEMATRIA_TESTING_LLVM_H_
#define THIRD_PARTY_GEMATRIA_GEMATRIA_TESTING_LLVM_H_

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "llvm/include/llvm/MC/MCInst.h"

namespace gematria {

// Creates a matcher for llvm::MCInst. The matcher succeeds when the opcode
// matches `opcode_matcher` and the list of all operand matches
// `operand_matchers`.
testing::Matcher<llvm::MCInst> IsMCInst(
    testing::Matcher<unsigned> opcode_matcher,
    testing::Matcher<std::vector<llvm::MCOperand>> operands_matcher);

// Creates a matcher for llvm::MCOperand that succeeds when the operand is an
// immediate value matching `value_matcher`.
testing::Matcher<llvm::MCOperand> IsImmediate(
    testing::Matcher<uint64_t> value_matcher);

// Creates a matcher for llvm::MCOperand that succeeds when the operand is a
// register operand and the register index matches `register_matcher`.
testing::Matcher<llvm::MCOperand> IsRegister(
    testing::Matcher<unsigned> register_matcher);

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_TESTING_LLVM_H_
