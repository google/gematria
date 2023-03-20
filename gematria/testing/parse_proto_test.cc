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

#include "gematria/testing/parse_proto.h"

#include "gematria/proto/canonicalized_instruction.pb.h"
#include "gtest/gtest.h"

namespace gematria {
namespace {

TEST(ParseTextProtoTest, Success) {
  const CanonicalizedOperandProto::AddressTuple proto = ParseTextProto(R"pb(
    base_register: 'RAX'
    index_register: 'RSI'
    displacement: -16
    scaling: 2
    segment: 'FS'
  )pb");
}

TEST(ParseTextProtoDeathTest, Failure) {
  CanonicalizedOperandProto::AddressTuple proto;
  EXPECT_DEATH(
      proto = ParseTextProto("this is not a valid proto in text format"),
      "Parsing proto from text format failed");
}

}  // namespace
}  // namespace gematria
