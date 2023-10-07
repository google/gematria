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

#ifndef THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_LLVM_TO_ABSL_H_
#define THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_LLVM_TO_ABSL_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace gematria {

absl::Status inline LlvmErrorToStatus(llvm::Error e) {
  if (!e) return absl::OkStatus();
  std::string ret;
  llvm::raw_string_ostream os(ret);
  os << e;
  return absl::InternalError(ret);
}

template <typename T>
absl::StatusOr<T> LlvmExpectedToStatusOr(llvm::Expected<T> expected) {
  if (expected) return std::move(*expected);
  return LlvmErrorToStatus(expected.takeError());
}
}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_LLVM_TO_ABSL_H_
