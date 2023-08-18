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

#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/source_location.h"
#include "file/base/path.h"
#include "gematria/basic_block/basic_block.h"
#include "gematria/basic_block/basic_block_protos.h"
#include "gematria/testing/parse_proto.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
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

// The path to a trained model based on the BasicBlockGraphBuilder code.
constexpr absl::string_view kMitModelFilePath =
    "google3/research/devtools/exegesis/gematria/granite/models/"
    "gb-token-mit-2022_12_02.tflite";

class GraphBuilderModelInferenceTest : public ::testing::Test {
 protected:
  std::unique_ptr<GraphBuilderModelInference> MakeModelInference(
      absl::string_view model_path) {
    const std::string full_model_path =
        file::JoinPath(absl::GetFlag(FLAGS_test_srcdir), model_path);
    tflite_model_ =
        tflite::FlatBufferModel::BuildFromFile(full_model_path.c_str());

    absl::StatusOr<std::unique_ptr<GraphBuilderModelInference>>
        inference_or_status =
            GraphBuilderModelInference::FromTfLiteModel(tflite_model_.get());
    ABSL_CHECK_OK(inference_or_status.status());

    return *std::move(inference_or_status);
  }

  std::unique_ptr<tflite::FlatBufferModel> tflite_model_;
};

bool AddBasicBlockToBatchFromTextProto(
    GraphBuilderModelInference& inference,
    absl::string_view basic_block_proto_pbtxt,
    absl::SourceLocation loc = absl::SourceLocation::current()) {
  const BasicBlock basic_block =
      BasicBlockFromProto(ParseTextProto(basic_block_proto_pbtxt));
  return inference.AddBasicBlockToBatch(basic_block);
}

TEST_F(GraphBuilderModelInferenceTest, SimpleRunInference) {
  std::unique_ptr<GraphBuilderModelInference> inference =
      MakeModelInference(kMitModelFilePath);
  ASSERT_TRUE(AddBasicBlockToBatchFromTextProto(*inference, kBasicBlocks[0]));
  EXPECT_THAT(inference->RunInference(),
              IsOkAndHolds(ElementsAre(BasicBlockOutputMatchers()[0])));
}

TEST_F(GraphBuilderModelInferenceTest, MultipleBlocksInBatch) {
  std::unique_ptr<GraphBuilderModelInference> inference =
      MakeModelInference(kMitModelFilePath);
  for (const absl::string_view basic_block_pbtxt : kBasicBlocks) {
    ASSERT_TRUE(
        AddBasicBlockToBatchFromTextProto(*inference, basic_block_pbtxt));
  }
  EXPECT_THAT(inference->RunInference(),
              IsOkAndHolds(ElementsAreArray(BasicBlockOutputMatchers())));
}

TEST_F(GraphBuilderModelInferenceTest, RunInferenceMultipleTimesWithNoReset) {
  std::unique_ptr<GraphBuilderModelInference> inference =
      MakeModelInference(kMitModelFilePath);
  const auto matchers = BasicBlockOutputMatchers();

  ASSERT_TRUE(AddBasicBlockToBatchFromTextProto(*inference, kBasicBlocks[0]));
  EXPECT_THAT(inference->RunInference(),
              IsOkAndHolds(ElementsAre(matchers[0])));

  ASSERT_TRUE(AddBasicBlockToBatchFromTextProto(*inference, kBasicBlocks[1]));
  EXPECT_THAT(inference->RunInference(),
              IsOkAndHolds(ElementsAreArray(matchers)));
}

TEST_F(GraphBuilderModelInferenceTest, RunInferenceMultipleTimesWithReset) {
  std::unique_ptr<GraphBuilderModelInference> inference =
      MakeModelInference(kMitModelFilePath);
  const auto matchers = BasicBlockOutputMatchers();

  ASSERT_TRUE(AddBasicBlockToBatchFromTextProto(*inference, kBasicBlocks[0]));
  EXPECT_THAT(inference->RunInference(),
              IsOkAndHolds(ElementsAre(matchers[0])));

  inference->Reset();

  ASSERT_TRUE(AddBasicBlockToBatchFromTextProto(*inference, kBasicBlocks[1]));
  EXPECT_THAT(inference->RunInference(),
              IsOkAndHolds(ElementsAre(matchers[1])));
}

TEST_F(GraphBuilderModelInferenceTest, UnknownToken) {
  std::unique_ptr<GraphBuilderModelInference> inference =
      MakeModelInference(kMitModelFilePath);

  EXPECT_TRUE(AddBasicBlockToBatchFromTextProto(*inference,
                                                kBasicBlockWithInvalidToken));
  // The basic block was not added, so the batch is empty.
  EXPECT_THAT(
      inference->RunInference(),
      IsOkAndHolds(ElementsAre(BasicBlockWithInvalidTokenOutputMatcher())));
}

TEST_F(GraphBuilderModelInferenceTest, UnknownTokenAfterValidBlock) {
  std::unique_ptr<GraphBuilderModelInference> inference =
      MakeModelInference(kMitModelFilePath);
  const auto matchers = BasicBlockOutputMatchers();

  EXPECT_TRUE(AddBasicBlockToBatchFromTextProto(*inference, kBasicBlocks[0]));
  EXPECT_TRUE(AddBasicBlockToBatchFromTextProto(*inference,
                                                kBasicBlockWithInvalidToken));
  // The second block was not added to the batch; we will se results only for
  // the first block.
  EXPECT_THAT(inference->RunInference(),
              IsOkAndHolds(ElementsAre(
                  matchers[0], BasicBlockWithInvalidTokenOutputMatcher())));
}

TEST_F(GraphBuilderModelInferenceTest, TwoNops) {
  constexpr absl::string_view kBlock = R"pb(
    canonicalized_instructions { mnemonic: "NOP" llvm_mnemonic: "NOOP" }
    canonicalized_instructions { mnemonic: "NOP" llvm_mnemonic: "NOOP" }
  )pb";
  std::unique_ptr<GraphBuilderModelInference> inference =
      MakeModelInference(kMitModelFilePath);
  EXPECT_TRUE(AddBasicBlockToBatchFromTextProto(*inference, kBlock));
  EXPECT_THAT(inference->RunInference(), IsOkAndHolds(ElementsAre(SizeIs(3))));
}

// An end-to-end example that is included as a go/g3doc-live-snippet in
// g3doc/research/devtools/exegesis/gematria/granite/g3doc/cc-api.md.
void EndToEndExample() {
  // Load the .tflite file. In this example, we load it from a file bundled with
  // the test. The TensorFlow Lite API also supports loading models from an
  // array of bytes.
  const std::string model_path =
      file::JoinPath(absl::GetFlag(FLAGS_test_srcdir), kMitModelFilePath);
  const std::unique_ptr<tflite::FlatBufferModel> tflite_model =
      tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

  // Create the inference class from the model. The parameters from the model
  // are baked into the .tflite file.
  const absl::StatusOr<std::unique_ptr<GraphBuilderModelInference>>
      inference_or_status =
          GraphBuilderModelInference::FromTfLiteModel(tflite_model.get());
  ABSL_CHECK_OK(inference_or_status.status());
  GraphBuilderModelInference& inference = **inference_or_status;

  constexpr int kNumBatches = 5;
  constexpr int kNumBlocksPerBatch = 10;
  absl::BitGen bitgen;
  for (int batch = 0; batch < kNumBatches; ++batch) {
    // Get some basic blocks to process. Replace GetSomeBlocks() with an actual
    // logic that determines which basic blocks get classified.
    std::vector<BasicBlock> basic_blocks;
    for (int block = 0; block < kNumBlocksPerBatch; ++block) {
      const size_t block_index =
          absl::Uniform(bitgen, 0ULL, std::size(kBasicBlocks));
      basic_blocks.push_back(
          BasicBlockFromProto(ParseTextProto(kBasicBlocks[block_index])));
    }

    // Create a batch with a couple of basic blocks. The call to Reset() is not
    // necessary for the first batch.
    inference.Reset();
    for (const BasicBlock& basic_block : basic_blocks) {
      inference.AddBasicBlockToBatch(basic_block);
    }
    const absl::StatusOr<std::vector<GraphBuilderModelInference::OutputType>>
        predictions = inference.RunInference();
    // In practice, a more nuanced error handling strategy would be appropriate.
    ABSL_CHECK_OK(predictions.status());
    // The number of outputs is always the same as the number of inputs.
    ABSL_CHECK_EQ(predictions->size(), basic_blocks.size());
    for (int block = 0; block < predictions->size(); ++block) {
      const auto& output = (*predictions)[block];
      ABSL_LOG(INFO) << "Basic block " << block;
      // All Gematria models are implemented as multi-task models, even if they
      // have just one output head (and `output` contains just a singel value).
      for (int task = 0; task < output.size(); ++task) {
        ABSL_LOG(INFO) << "  task " << task << ": " << output[task];
      }
    }
  }
}

TEST(GraphBuilderG3DocTest, EndToEndExample) { EndToEndExample(); }

}  // namespace
}  // namespace gematria
