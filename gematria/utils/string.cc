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

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gematria {
namespace {

constexpr int kInvalidHexDigit = 256;

int ParseHexDigit(char digit) {
  if (digit >= '0' && digit <= '9') {
    return digit - '0';
  }
  if (digit >= 'a' && digit <= 'f') {
    return digit - 'a' + 10;
  }
  if (digit >= 'A' && digit <= 'F') {
    return digit - 'A' + 10;
  }
  return kInvalidHexDigit;
}

bool IsAsciiWhitespace(char c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

}  // namespace

std::optional<std::vector<uint8_t>> ParseHexString(
    std::string_view hex_string) {
  if (hex_string.size() % 2 != 0) {
    return std::nullopt;
  }
  std::vector<uint8_t> res;
  while (!hex_string.empty()) {
    const int hex_value =
        (ParseHexDigit(hex_string[0]) << 4) + ParseHexDigit(hex_string[1]);
    if (hex_value >= 256) {
      return std::nullopt;
    }
    res.push_back(hex_value);
    hex_string.remove_prefix(2);
  }

  return res;
}

std::vector<std::string> StrSplitAsCopy(std::string_view text, char separator) {
  std::vector<std::string> splits;
  std::string_view::size_type last_separator = 0;
  std::string_view::size_type next_separator = text.find(separator);
  while (next_separator != std::string_view::npos) {
    splits.emplace_back(
        text.substr(last_separator, next_separator - last_separator));
    last_separator = next_separator + 1;
    next_separator = text.find(separator, last_separator);
  }
  splits.emplace_back(text.substr(last_separator));
  return splits;
}

void StripAsciiWhitespace(std::string* text) {
  assert(text != nullptr);
  auto begin = text->begin();
  auto end = text->end();
  while (begin != end && IsAsciiWhitespace(*begin)) ++begin;
  while (begin != end && IsAsciiWhitespace(*(end - 1))) --end;
  text->assign(begin, end);
}

}  // namespace gematria
