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

#include "gematria/llvm/llvm_to_absl.h"

#include <utility>

#include "absl/status/status.h"
#include "gematria/testing/matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

namespace gematria {
namespace {

TEST(LlvmErrorToStatus, Success) {
  llvm::Error e = llvm::Error::success();
  EXPECT_OK(LlvmErrorToStatus(std::move(e)));
}

TEST(LlvmErrorToStatus, Failure) {
  llvm::Error e =
      llvm::make_error<llvm::StringError>(llvm::errc::io_error, "broken!");
  EXPECT_THAT(
      LlvmErrorToStatus(std::move(e)),
      StatusIs(absl::StatusCode::kInternal, testing::HasSubstr("broken!")));
}

TEST(LlvmExpectedToStatusOr, Success) {
  llvm::Expected<int> e = 42;
  EXPECT_THAT(LlvmExpectedToStatusOr(std::move(e)), IsOkAndHolds(42));
}

TEST(LlvmExpectedToStatusOr, Failure) {
  llvm::Expected<int> e =
      llvm::make_error<llvm::StringError>(llvm::errc::io_error, "broken!");
  EXPECT_THAT(
      LlvmExpectedToStatusOr(std::move(e)),
      StatusIs(absl::StatusCode::kInternal, testing::HasSubstr("broken!")));
}
}  // namespace
}  // namespace gematria
