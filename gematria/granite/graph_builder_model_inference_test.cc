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

#include "gematria/granite/graph_builder_model_inference.h"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/source_location.h"
#include "file/base/path.h"
#include "gematria/basic_block/basic_block.h"
#include "gematria/basic_block/basic_block_protos.h"
#include "gematria/llvm/llvm_to_absl.h"
#include "gematria/testing/parse_proto.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "tensorflow/lite/model_builder.h"
#include "testing/base/public/googletest.h"

namespace gematria {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::FloatNear;
using ::testing::Matcher;
using ::testing::SizeIs;
using ::testing::status::IsOkAndHolds;

// The tolerance used when matching the output of the model with the expected
// results. Note that the output of the model is always the same, the tolerance
// here is used to make the floating point literals in the matchers shorter and
// more readable.
constexpr float kTolerance = 1e-3;

void AbortOnError(llvm::Error error) {
  if (error) {
    llvm::dbgs() << "Fatal error: " << error << "\n";
    std::abort();
  }
}

// Basic blocks used in the tests.
constexpr absl::string_view kBasicBlocks[] = {
    R"pb(
      canonicalized_instructions {
        mnemonic: "MOV"
        llvm_mnemonic: "MOV64rr"
        output_operands { register_name: "RSI" }
        input_operands { register_name: "RBX" }
      }
      canonicalized_instructions {
        mnemonic: "MOV"
        llvm_mnemonic: "MOV64rr"
        output_operands { register_name: "RDX" }
        input_operands { register_name: "RAX" }
      }
      canonicalized_instructions {
        mnemonic: "MOV"
        llvm_mnemonic: "MOV64rr"
        output_operands { register_name: "RDI" }
        input_operands { register_name: "R15" }
      })pb",
    R"pb(
      canonicalized_instructions: {
        mnemonic: "LEA"
        llvm_mnemonic: "LEA64r"  # size=6
        output_operands: { register_name: "RDI" }
        input_operands: {
          address: { base_register: "RBX" displacement: 8 scaling: 1 }
        }
      })pb",
};

// A basic block that uses an invalid token `SomeUnknownInstruction`. This token
// should be recognized by the model and handled according to the way how the
// model was constructed (i.e. the basic block should be ignored or the unknown
// token should be replaced with a special replacement token).
constexpr absl::string_view kBasicBlockWithInvalidToken = R"pb(
  canonicalized_instructions: {
    mnemonic: "SomeUnknownInstruction"
    llvm_mnemonic: "LEA64r"  # size=6
    output_operands: { register_name: "RDI" }
    input_operands: {
      address: { base_register: "RBX" displacement: 8 scaling: 1 }
    }
  })pb";

// Returns matchers for the expected outputs of the inference for the basic
// blocks from kBasicBlocks.
std::vector<Matcher<GraphBuilderModelInference::OutputType>>
BasicBlockOutputMatchers() {
  return {
      ElementsAre(FloatNear(88.2726, kTolerance),
                  FloatNear(101.896561, kTolerance),
                  FloatNear(86.6651535, kTolerance)),
      ElementsAre(FloatNear(37.455822, kTolerance),
                  FloatNear(35.8950844, kTolerance),
                  FloatNear(35.4254227, kTolerance)),
  };
}

// Returns matchers for the expected output of the inference for
// kBasicBlockWithInvalidToken.
Matcher<GraphBuilderModelInference::OutputType>
BasicBlockWithInvalidTokenOutputMatcher() {
  return ElementsAre(FloatNear(37.6083946, kTolerance),
                     FloatNear(36.3878174, kTolerance),
                     FloatNear(36.2444115, kTolerance));
}

}  // namespace
}  // namespace gematria
