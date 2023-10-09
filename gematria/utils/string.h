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

#ifndef THIRD_PARTY_GEMATRIA_GEMATRIA_UTILS_STRING_H_
#define THIRD_PARTY_GEMATRIA_GEMATRIA_UTILS_STRING_H_

#include <cstdint>
#include <iomanip>
#include <ios>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace gematria {

// Parses a sequence of bytes from a hexadecimal format. The input string must
// contain an even number of hex digits; each pair of hex digits is interpreted
// as one byte in hex format; for example the string "ABCDEF" translates to the
// vector {0xAB, 0xCD, 0xEF}.
// Returns an error when the string has an odd length or it contains characters
// that are not hex digits.
std::optional<std::vector<uint8_t>> ParseHexString(std::string_view hex_string);

// Formats `bytes` as a hex string that can be parsed with ParseHexString().
inline std::string FormatAsHexString(std::string_view bytes) {
  std::stringstream out;
  out << std::hex;
  for (const auto byte : bytes) {
    out << std::setfill('0') << std::setw(2) << static_cast<int>(byte);
  }
  return out.str();
}

// Splits `text` by `separator`. Note that the returned split string is a copy
// independent of the input text.
std::vector<std::string> StrSplitAsCopy(std::string_view text, char separator);

// Strips the leading and trailing whitespace from `text` and returns the
// modified text.
void StripAsciiWhitespace(std::string* text);

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_UTILS_STRING_H_
