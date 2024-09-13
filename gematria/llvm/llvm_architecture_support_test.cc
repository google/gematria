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
#include <utility>

#include "absl/status/status.h"
#include "gematria/llvm/llvm_to_absl.h"
#include "gematria/testing/matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace gematria {
namespace {

TEST(LlvmArchitectureSupportTest, X86_64) {
  std::unique_ptr<LlvmArchitectureSupport> x86_64 =
      LlvmArchitectureSupport::X86_64();
  ASSERT_NE(x86_64, nullptr);
  EXPECT_EQ(x86_64->target_machine().getTargetTriple().getArchName(), "x86_64");
}

TEST(LlvmArchitectureSupportTest, FromTriple) {
  auto x86_64_or_status = LlvmExpectedToStatusOr(
      LlvmArchitectureSupport::FromTriple("x86_64", "", ""));
  ASSERT_OK(x86_64_or_status);
  std::unique_ptr<LlvmArchitectureSupport> x86_64 =
      std::move(x86_64_or_status).value();

  ASSERT_NE(x86_64, nullptr);
  EXPECT_EQ(x86_64->target_machine().getTargetTriple().getArchName(), "x86_64");
}

TEST(LlvmArchitectureSupportTest, FromTriple_Invalid) {
  auto x86_64_or_status =
      LlvmExpectedToStatusOr(LlvmArchitectureSupport::FromTriple(
          "an_architecture_that_does_not_exist", "", ""));
  EXPECT_THAT(x86_64_or_status, StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace gematria
