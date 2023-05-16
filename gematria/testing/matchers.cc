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

#include "gematria/testing/matchers.h"

#include <string>
#include <string_view>

#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "gmock/gmock.h"

namespace gematria {
namespace {

using ::testing::Matcher;
using ::testing::ResultOf;

}  // namespace

bool EqualsNormalizingWhitespace(std::string_view left,
                                 std::string_view right) {
  left = absl::StripAsciiWhitespace(left);
  right = absl::StripAsciiWhitespace(right);
  while (!left.empty() && !right.empty()) {
    const bool left_has_space = absl::ascii_isspace(left.front());
    const bool right_has_space = absl::ascii_isspace(right.front());
    if (left_has_space != right_has_space) return false;
    if (left_has_space) {
      left = absl::StripLeadingAsciiWhitespace(left);
      right = absl::StripLeadingAsciiWhitespace(right);
    } else {
      if (left.front() != right.front()) return false;
      left.remove_prefix(1);
      right.remove_prefix(1);
    }
  }
  return left.empty() && right.empty();
}

Matcher<std::string> EqualsNormalizingWhitespace(std::string other) {
  std::string matcher_name =
      absl::StrFormat("EqualsNormalizingWhitespace(\"%s\")", other);
  return ResultOf(
      std::move(matcher_name),
      [left = std::move(other)](const std::string& right) {
        return EqualsNormalizingWhitespace(left, right);
      },
      testing::IsTrue());
}

namespace internal {
namespace {
using ::google::protobuf::Descriptor;
using ::google::protobuf::DescriptorPool;
using ::google::protobuf::FieldDescriptor;
}  // namespace

void AddIgnoredFieldsToDifferencer(
    const Descriptor* descriptor,
    const std::vector<std::string>& ignored_field_names,
    ::google::protobuf::util::MessageDifferencer* differencer) {
  CHECK(descriptor != nullptr);
  CHECK(differencer != nullptr);
  const DescriptorPool* const pool = descriptor->file()->pool();
  for (const std::string& field_name : ignored_field_names) {
    const FieldDescriptor* const field = pool->FindFieldByName(field_name);
    if (field == nullptr) {
      ADD_FAILURE() << "Field \"" << field_name << "\" was not found.";
      continue;
    }
    differencer->IgnoreField(field);
  }
}

}  // namespace internal
}  // namespace gematria
