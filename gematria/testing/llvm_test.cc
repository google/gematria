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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "lib/Target/X86/MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"

namespace gematria {
namespace {

using ::testing::_;
using ::testing::AnyOf;
using ::testing::ElementsAre;
using ::testing::Gt;
using ::testing::IsEmpty;
using ::testing::Value;

TEST(IsMCInstTest, MatchOnlyOpcode) {
  EXPECT_TRUE(Value(llvm::MCInst(llvm::MCInstBuilder(llvm::X86::NOOP)),
                    IsMCInst(llvm::X86::NOOP, _)));
  EXPECT_FALSE(Value(llvm::MCInst(llvm::MCInstBuilder(llvm::X86::NOOP)),
                     IsMCInst(llvm::X86::ADD32rr, _)));
  EXPECT_FALSE(Value(llvm::MCInst(llvm::MCInstBuilder(llvm::X86::MOV32rr)
                                      .addReg(llvm::X86::EAX)
                                      .addReg(llvm::X86::EBX)),
                     IsMCInst(llvm::X86::NOOP, _)));
}

TEST(IsMCInstTest, MatchOpcodeAndOperands) {
  EXPECT_TRUE(Value(llvm::MCInst(llvm::MCInstBuilder(llvm::X86::NOOP)),
                    IsMCInst(llvm::X86::NOOP, IsEmpty())));
  EXPECT_FALSE(Value(
      llvm::MCInst(llvm::MCInstBuilder(llvm::X86::NOOP)),
      IsMCInst(llvm::X86::NOOP, ElementsAre(IsRegister(llvm::X86::RAX)))));
  EXPECT_TRUE(Value(
      llvm::MCInst(llvm::MCInstBuilder(llvm::X86::MOV32rr)
                       .addReg(llvm::X86::EAX)
                       .addReg(llvm::X86::EBX)),
      IsMCInst(llvm::X86::MOV32rr, ElementsAre(IsRegister(llvm::X86::EAX),
                                               IsRegister(llvm::X86::EBX)))));
  EXPECT_FALSE(Value(llvm::MCInst(llvm::MCInstBuilder(llvm::X86::MOV32rr)
                                      .addReg(llvm::X86::EAX)
                                      .addReg(llvm::X86::EBX)),
                     IsMCInst(llvm::X86::MOV32rr, IsEmpty())));
  EXPECT_FALSE(Value(
      llvm::MCInst(llvm::MCInstBuilder(llvm::X86::MOV32rr)
                       .addReg(llvm::X86::EAX)
                       .addReg(llvm::X86::EBX)),
      IsMCInst(llvm::X86::MOV32rr, ElementsAre(IsRegister(llvm::X86::EAX),
                                               IsRegister(llvm::X86::ECX)))));
}

TEST(IsImmediateTest, NotAnImmediate) {
  EXPECT_FALSE(
      Value(llvm::MCOperand::createReg(llvm::X86::RIP), IsImmediate(_)));
  EXPECT_FALSE(Value(llvm::MCOperand::createSFPImm(0.0f), IsImmediate(_)));
  EXPECT_FALSE(Value(llvm::MCOperand::createSFPImm(1.0f), IsImmediate(_)));
}

TEST(IsImmediateTest, IsAnImmediate) {
  EXPECT_TRUE(Value(llvm::MCOperand::createImm(123), IsImmediate(_)));
  EXPECT_TRUE(Value(llvm::MCOperand::createImm(111), IsImmediate(111)));
  EXPECT_TRUE(
      Value(llvm::MCOperand::createImm(9999999), IsImmediate(Gt(1000))));
}

TEST(IsRegisterTest, NotARegister) {
  EXPECT_FALSE(Value(llvm::MCOperand::createImm(123), IsRegister(_)));
  EXPECT_FALSE(Value(llvm::MCOperand::createSFPImm(123.0f), IsRegister(_)));
  EXPECT_FALSE(Value(llvm::MCOperand::createDFPImm(123.0), IsRegister(_)));
}

TEST(IsRegisterTest, IsSpecificRegister) {
  EXPECT_TRUE(Value(llvm::MCOperand::createReg(llvm::X86::RAX),
                    IsRegister(llvm::X86::RAX)));
  EXPECT_TRUE(Value(llvm::MCOperand::createReg(llvm::X86::CL),
                    IsRegister(llvm::X86::CL)));
  EXPECT_FALSE(Value(llvm::MCOperand::createReg(llvm::X86::RAX),
                     IsRegister(llvm::X86::AL)));
  EXPECT_FALSE(Value(llvm::MCOperand::createReg(llvm::X86::RAX),
                     IsRegister(llvm::X86::RSI)));
}

TEST(IsRegisterTest, IsRegisterWithMatcher) {
  EXPECT_TRUE(Value(llvm::MCOperand::createReg(llvm::X86::RBX), IsRegister(_)));
  EXPECT_TRUE(
      Value(llvm::MCOperand::createReg(llvm::X86::RCX),
            IsRegister(AnyOf(llvm::X86::RAX, llvm::X86::RBX, llvm::X86::RCX))));
}

}  // namespace
}  // namespace gematria
