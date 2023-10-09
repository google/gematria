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

#include "gematria/utils/string.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace gematria {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Optional;

TEST(ParseHexStringTest, SomeValidHexStrings) {
  const struct {
    std::string_view hex_string;
    std::vector<uint8_t> expected_bytes;
  } kTestCases[] = {
      {"", {}},
      {"01", {0x1}},
      {"abcdef", {0xab, 0xcd, 0xef}},
      {"AbCdEf0123456789AbCdEf",
       {0xAB, 0xCD, 0xEF, 0x1, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF}},
  };
  for (const auto& test_case : kTestCases) {
    EXPECT_THAT(ParseHexString(test_case.hex_string),
                Optional(ElementsAreArray(test_case.expected_bytes)));
  }
}

TEST(ParseHexStringTest, SomeInvalidHexStrings) {
  static constexpr std::string_view kTestCases[] = {
      // Odd length.
      "000",
      // Invalid character in the string.
      "f00bar"};
  for (const std::string_view& test_case : kTestCases) {
    EXPECT_EQ(ParseHexString(test_case), std::nullopt);
  }
}

TEST(FormatAsHexStringTest, EmptySpan) { EXPECT_EQ(FormatAsHexString(""), ""); }

TEST(FormatAsHexStringTest, NonEmptySpan) {
  static constexpr char kBytes[] = {0x1, 0x23, 0xAB, 0xCD};
  static constexpr std::string_view kBytesAsStringView(kBytes,
                                                       std::size(kBytes));
  EXPECT_EQ(FormatAsHexString(kBytesAsStringView), "0123abcd");
}

TEST(StrSplitAsCopyTest, EmptyText) {
  EXPECT_THAT(StrSplitAsCopy("", ';'), ElementsAre(""));
}

TEST(StrSplitAsCopyTest, NoSeparator) {
  EXPECT_THAT(StrSplitAsCopy("foobar", ';'), ElementsAre("foobar"));
}

TEST(StrSplitAsCopyTest, Separators) {
  EXPECT_THAT(StrSplitAsCopy(";foo;;bar;baz", ';'),
              ElementsAre("", "foo", "", "bar", "baz"));
}

TEST(StrSplitAsCopyTest, SeparatorAtEnd) {
  EXPECT_THAT(StrSplitAsCopy("foo;bar;", ';'), ElementsAre("foo", "bar", ""));
}

TEST(StripAsciiWhitespaceTest, EmptyText) {
  std::string text;
  StripAsciiWhitespace(&text);
  EXPECT_EQ(text, "");
}

TEST(StripAsciiWhitespaceTest, NoLeadingOrTrailingWhitespace) {
  std::string text = "foo \t\r\nbar";
  StripAsciiWhitespace(&text);
  EXPECT_EQ(text, "foo \t\r\nbar");
}

TEST(StripAsciiWhitespaceTest, LeadingAndTrailingWhitespace) {
  std::string text = " \t\r\rfoobar\n\n\r \t";
  StripAsciiWhitespace(&text);
  EXPECT_EQ(text, "foobar");
}

TEST(StripAsciiWhitespaceTest, OnlyWhitespace) {
  std::string text = "\t\r\t           \n\n\n\r";
  StripAsciiWhitespace(&text);
  EXPECT_EQ(text, "");
}

}  // namespace
}  // namespace gematria
