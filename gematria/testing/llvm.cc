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

#include "gematria/testing/llvm.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/include/llvm/MC/MCInst.h"

namespace gematria {

using testing::AllOf;
using testing::IsTrue;
using testing::Property;
using testing::ResultOf;

std::vector<llvm::MCOperand> GetAllOperands(const llvm::MCInst& mc_inst) {
  const int num_operands = mc_inst.getNumOperands();
  std::vector<llvm::MCOperand> operands;
  operands.reserve(num_operands);
  for (int i = 0; i < mc_inst.getNumOperands(); ++i) {
    operands.push_back(mc_inst.getOperand(i));
  }
  return operands;
}

testing::Matcher<llvm::MCInst> IsMCInst(
    testing::Matcher<unsigned> opcode_matcher,
    testing::Matcher<std::vector<llvm::MCOperand>> operands_matcher) {
  return AllOf(Property("getOpcode", &llvm::MCInst::getOpcode, opcode_matcher),
               ResultOf("operands", &GetAllOperands, operands_matcher));
}

testing::Matcher<llvm::MCOperand> IsImmediate(
    testing::Matcher<uint64_t> value_matcher) {
  return AllOf(Property("isImm", &llvm::MCOperand::isImm, IsTrue()),
               Property("getImm", &llvm::MCOperand::getImm, value_matcher));
}

testing::Matcher<llvm::MCOperand> IsRegister(
    testing::Matcher<unsigned> register_matcher) {
  return AllOf(Property("isReg", &llvm::MCOperand::isReg, IsTrue()),
               Property("getReg", &llvm::MCOperand::getReg, register_matcher));
}

}  // namespace gematria
