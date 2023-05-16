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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace gematria {
namespace {

using ::testing::Value;

TEST(EqualsNormalizingWhitespaceTest, Function) {
  EXPECT_TRUE(EqualsNormalizingWhitespace("foo", "foo"));
  EXPECT_TRUE(EqualsNormalizingWhitespace("\r\rfoo    ", "foo"));
  EXPECT_TRUE(EqualsNormalizingWhitespace("foo", "\t\tfoo    "));
  EXPECT_TRUE(EqualsNormalizingWhitespace("foo bar", "foo     \n\r    bar"));

  EXPECT_FALSE(EqualsNormalizingWhitespace("foo", "bar"));
  EXPECT_FALSE(EqualsNormalizingWhitespace("foo bar", "foobar"));
  EXPECT_FALSE(EqualsNormalizingWhitespace("foo bar", "foo foo"));
}

TEST(EqualsNormalizingWhitespaceTest, Matcher) {
  EXPECT_TRUE(Value("foo", EqualsNormalizingWhitespace("foo")));
  EXPECT_TRUE(Value("foo", EqualsNormalizingWhitespace("\r\rfoo\t   ")));
  EXPECT_TRUE(Value("\t\tfoo    ", EqualsNormalizingWhitespace("foo")));
  EXPECT_TRUE(
      Value("foo     \n\r    bar", EqualsNormalizingWhitespace("foo bar")));

  EXPECT_FALSE(Value("bar", EqualsNormalizingWhitespace("foo")));
  EXPECT_FALSE(Value("foobar", EqualsNormalizingWhitespace("foo bar")));
  EXPECT_FALSE(Value("foo foo", EqualsNormalizingWhitespace("foo bar")));
}

}  // namespace
}  // namespace gematria
