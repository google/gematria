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

#include "gematria/tflite/unsorted_segment_sum_op.h"

#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace gematria {
namespace {

using ::testing::ElementsAre;

class UnsortedSegmentSumOpModel : public tflite::SingleOpModel {
 public:
  UnsortedSegmentSumOpModel(const tflite::TensorData& data,
                            const tflite::TensorData& segment_ids,
                            const tflite::TensorData& num_segments,
                            const tflite::TensorData& output) {
    data_id_ = AddInput(data);
    segment_ids_id_ = AddInput(segment_ids);
    num_segments_id_ = AddInput(num_segments);
    output_id_ = AddOutput(output);
    SetCustomOp(tflite::string(kUnsortedSegmentSumOpName), {},
                RegisterUnsortedSegmentSumOp);
    BuildInterpreter({GetShape(data_id_), GetShape(segment_ids_id_),
                      GetShape(num_segments_id_)});
  }

  int data() const { return data_id_; }
  int segment_ids() const { return segment_ids_id_; }
  int num_segments() const { return num_segments_id_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_id_); }
  std::vector<int32_t> GetOutputShape() { return GetTensorShape(output_id_); }

 protected:
  int data_id_;
  int segment_ids_id_;
  int num_segments_id_;
  int output_id_;
};

TEST(UnsortedSegmentSumOpModelTest, Trivial1DMatrix) {
  UnsortedSegmentSumOpModel model(
      /* data = */ {tflite::TensorType_FLOAT32, {6}},
      /* segment_ids = */ {tflite::TensorType_INT32, {6}},
      /* num_segments = */ {tflite::TensorType_INT32, {}},
      /* output = */ {tflite::TensorType_FLOAT32, {4}});
  model.PopulateTensor<float>(model.data(),
                              {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1, 2, 0, 0, 2});
  model.PopulateTensor<int32_t>(model.num_segments(), {4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  // The last element is required by `num_segments`, but it is not used in any
  // of the segments, so it remains at zero.
  EXPECT_THAT(model.GetOutput(), ElementsAre(19.0f, 5.0f, 15.0f, 0.0f));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
}

TEST(UnsortedSegmentSumOpModelTest, Flat2DMatrix) {
  UnsortedSegmentSumOpModel model(
      /* data = */ {tflite::TensorType_FLOAT32, {6, 1}},
      /* segment_ids = */ {tflite::TensorType_INT32, {6}},
      /* num_segments = */ {tflite::TensorType_INT32, {}},
      /* output = */ {tflite::TensorType_FLOAT32, {4, 1}});
  model.PopulateTensor<float>(model.data(),
                              {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1, 2, 0, 0, 2});
  model.PopulateTensor<int32_t>(model.num_segments(), {4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  // The last element is required by `num_segments`, but it is not used in any
  // of the segments, so it remains at zero.
  EXPECT_THAT(model.GetOutput(), ElementsAre(19.0f, 5.0f, 15.0f, 0.0f));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 1));
}

TEST(UnsortedSegmentSumOpModelTest, Normal2DMatrix) {
  UnsortedSegmentSumOpModel model(
      /* data = */ {tflite::TensorType_FLOAT32, {3, 2}},
      /* segment_ids = */ {tflite::TensorType_INT32, {3}},
      /* num_segments = */ {tflite::TensorType_INT32, {}},
      /* output = */ {tflite::TensorType_FLOAT32, {4, 2}});
  model.PopulateTensor<float>(model.data(),
                              {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 0, 2});
  model.PopulateTensor<int32_t>(model.num_segments(), {4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  // The last element is required by `num_segments`, but it is not used in any
  // of the segments, so it remains at zero.
  EXPECT_THAT(model.GetOutput(),
              ElementsAre(10.0f, 12.0f, 0.0f, 0.0f, 8.0f, 9.0f, 0.0f, 0.0f));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 2));
}

TEST(UnsortedSegmentSumOpModelTest, With2DSegmentIdMatrix) {
  UnsortedSegmentSumOpModel model(
      /* data = */ {tflite::TensorType_FLOAT32, {3, 2}},
      /* segment_ids = */ {tflite::TensorType_INT32, {3, 2}},
      /* num_segments = */ {tflite::TensorType_INT32, {}},
      /* output = */ {tflite::TensorType_FLOAT32, {4}});
  model.PopulateTensor<float>(model.data(),
                              {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 0, 2, 1, 2, 0});
  model.PopulateTensor<int32_t>(model.num_segments(), {4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  // The last element is required by `num_segments`, but it is not used in any
  // of the segments, so it remains at zero.
  EXPECT_THAT(model.GetOutput(), ElementsAre(18.0f, 7.0f, 14.0f, 0.0f));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
}

TEST(UnsortedSegmentSumOpModelTest, NotMatchingShapes) {
  // The shape of the segment IDs tensor does not match the shape of the data
  // tensor (they have a different size in the first dimension).
  UnsortedSegmentSumOpModel model(
      /* data = */ {tflite::TensorType_FLOAT32, {3}},
      /* segment_ids = */ {tflite::TensorType_INT32, {4}},
      /* num_segments = */ {tflite::TensorType_INT32, {}},
      /* output = */ {tflite::TensorType_FLOAT32, {4}});
  model.PopulateTensor<float>(model.data(), {4.0f, 5.0f, 6.0f});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 0, 2, 1});
  model.PopulateTensor<int32_t>(model.num_segments(), {3});

  EXPECT_EQ(model.Invoke(), kTfLiteError);
}

TEST(UnsortedSegmentSumOpModelTest, SegmentIdOVerflow) {
  UnsortedSegmentSumOpModel model(
      /* data = */ {tflite::TensorType_FLOAT32, {3}},
      /* segment_ids = */ {tflite::TensorType_INT32, {3}},
      /* num_segments = */ {tflite::TensorType_INT32, {}},
      /* output = */ {tflite::TensorType_FLOAT32, {3}});
  model.PopulateTensor<float>(model.data(), {4.0f, 5.0f, 6.0f});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 0, 10});
  model.PopulateTensor<int32_t>(model.num_segments(), {3});

  EXPECT_EQ(model.Invoke(), kTfLiteError);
}

TEST(UnsortedSegmentSumOpModelTest, NegativeSegmentId) {
  UnsortedSegmentSumOpModel model(
      /* data = */ {tflite::TensorType_FLOAT32, {3}},
      /* segment_ids = */ {tflite::TensorType_INT32, {3}},
      /* num_segments = */ {tflite::TensorType_INT32, {}},
      /* output = */ {tflite::TensorType_FLOAT32, {3}});
  model.PopulateTensor<float>(model.data(), {4.0f, 5.0f, 6.0f});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, -1, 2});
  model.PopulateTensor<int32_t>(model.num_segments(), {3});

  EXPECT_EQ(model.Invoke(), kTfLiteError);
}

}  // namespace
}  // namespace gematria
