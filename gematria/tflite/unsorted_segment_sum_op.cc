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

#include <algorithm>
#include <cstdint>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace gematria {
namespace {

// The indices of the inputs and outputs of the op.
constexpr int kInputDataTensor = 0;
constexpr int kInputSegmentIdsTensor = 1;
constexpr int kInputNumSegmentsTensor = 2;
constexpr int kNumInputs = 3;

constexpr int kOutputTensor = 0;
constexpr int kNumOutputs = 1;

// Resizes the output tensor based on the sizes of the input tensors.
// Requires that:
//   - the shape of `segment_ids_tensor` is a prefix of the shape of
//     `data_tensor`; they can both have the same shape.
//   - the value of `num_segments_tensor` can be read.
// The shape of the output is computed as: num_segments x output_data_dimensions
// where `num_segments` is the value of `num_segments_tensor`, and
// `output_data_dimensions` is a tuple of dimensions of `data_tensor` without
// the dimensions shared with `segment_ids_tensor`.
//
// For example, when `data_tensor` has shape (10, 3, 5, 4), `segment_ids_tensor`
// has shape (10, 3), and `num_segments_tensor` contains 7, the output shape
// will be (7, 5, 4).
TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const TfLiteTensor* data_tensor,
                                const TfLiteTensor* segment_ids_tensor,
                                const TfLiteTensor* num_segments_tensor,
                                TfLiteTensor* output_tensor) {
  const int num_segments =
      tflite::GetTensorData<int32_t>(num_segments_tensor)[0];
  TF_LITE_ENSURE(context, num_segments > 0);

  const int num_data_dimensions = tflite::NumDimensions(data_tensor);
  const int num_segment_ids_dimensions =
      tflite::NumDimensions(segment_ids_tensor);
  const int num_output_dimensions =
      num_data_dimensions - num_segment_ids_dimensions + 1;
  TF_LITE_ENSURE(context, num_output_dimensions > 0);

  // Check that the shape of the segment IDs tensor is a "prefix" of the shape
  // of the data tensor.
  for (int i = 0; i < num_segment_ids_dimensions; ++i) {
    TF_LITE_ENSURE_EQ(context, data_tensor->dims->data[i],
                      segment_ids_tensor->dims->data[i]);
  }

  // Create the shape of the output tensor.
  TfLiteIntArray* const output_size =
      TfLiteIntArrayCreate(num_output_dimensions);
  output_size->data[0] = num_segments;
  for (int i = 1; i < num_output_dimensions; ++i) {
    output_size->data[i] =
        data_tensor->dims->data[num_segment_ids_dimensions + i - 1];
  }
  return context->ResizeTensor(context, output_tensor, output_size);
}

TfLiteStatus UnsortedSegmentSumPrepare(TfLiteContext* context,
                                       TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), kNumInputs);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), kNumOutputs);

  const TfLiteTensor* data_tensor = nullptr;
  TF_LITE_ENSURE_OK(
      context,
      tflite::GetInputSafe(context, node, kInputDataTensor, &data_tensor));
  const TfLiteTensor* segment_ids_tensor = nullptr;
  TF_LITE_ENSURE_OK(context,
                    tflite::GetInputSafe(context, node, kInputSegmentIdsTensor,
                                         &segment_ids_tensor));
  const TfLiteTensor* num_segments_tensor = nullptr;
  TF_LITE_ENSURE_OK(context,
                    tflite::GetInputSafe(context, node, kInputNumSegmentsTensor,
                                         &num_segments_tensor));

  // TODO(ondrasej): Consider supporting additional data types.
  TF_LITE_ENSURE_EQ(context, data_tensor->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, segment_ids_tensor->type, kTfLiteInt32);

  const int num_data_dimensions = tflite::NumDimensions(data_tensor);
  const int num_segment_ids_dimensions =
      tflite::NumDimensions(segment_ids_tensor);
  TF_LITE_ENSURE(context, num_data_dimensions >= num_segment_ids_dimensions);

  TF_LITE_ENSURE_EQ(context, num_segments_tensor->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(num_segments_tensor), 0);
  TF_LITE_ENSURE_EQ(context, num_segments_tensor->bytes, sizeof(int32_t));

  TfLiteTensor* output_tensor = nullptr;
  TF_LITE_ENSURE_OK(context, tflite::GetOutputSafe(context, node, kOutputTensor,
                                                   &output_tensor));
  TF_LITE_ENSURE_EQ(context, output_tensor->type, kTfLiteFloat32);

  // TODO(ondrasej): This condition is safe, but perhaps we could find a weaker
  // condition for allocating the output during the `prepare` phase.
  if (!tflite::IsConstantTensor(num_segments_tensor) ||
      !tflite::IsConstantTensor(data_tensor) ||
      !tflite::IsConstantTensor(segment_ids_tensor)) {
    tflite::SetTensorToDynamic(output_tensor);
    return kTfLiteOk;
  }
  return ResizeOutputTensor(context, /* data_tensor = */ data_tensor,
                            /* segment_ids_tensor = */ segment_ids_tensor,
                            /* num_segments_tensor = */ num_segments_tensor,
                            /* output_tensor = */ output_tensor);
}

TfLiteStatus UnsortedSegmentSumInvoke(TfLiteContext* context,
                                      TfLiteNode* node) {
  const TfLiteTensor* data_tensor = nullptr;
  TF_LITE_ENSURE_OK(
      context,
      tflite::GetInputSafe(context, node, kInputDataTensor, &data_tensor));
  const TfLiteTensor* segment_ids_tensor = nullptr;
  TF_LITE_ENSURE_OK(context,
                    tflite::GetInputSafe(context, node, kInputSegmentIdsTensor,
                                         &segment_ids_tensor));
  const TfLiteTensor* num_segments_tensor = nullptr;
  TF_LITE_ENSURE_OK(context,
                    tflite::GetInputSafe(context, node, kInputNumSegmentsTensor,
                                         &num_segments_tensor));
  TfLiteTensor* output_tensor = nullptr;
  TF_LITE_ENSURE_OK(context, tflite::GetOutputSafe(context, node, kOutputTensor,
                                                   &output_tensor));

  if (tflite::IsDynamicTensor(output_tensor)) {
    TF_LITE_ENSURE_OK(
        context,
        ResizeOutputTensor(context, /* data_tensor = */ data_tensor,
                           /* segment_ids_tensor = */ segment_ids_tensor,
                           /* num_segments_tensor = */ num_segments_tensor,
                           /* output_tensor = */ output_tensor));
  }
  const tflite::RuntimeShape data_shape = tflite::GetTensorShape(data_tensor);
  const int data_flat_size = data_shape.FlatSize();
  const auto* const data = tflite::GetTensorData<float>(data_tensor);

  const tflite::RuntimeShape index_shape =
      tflite::GetTensorShape(segment_ids_tensor);
  const int index_flat_size = index_shape.FlatSize();
  const auto* const index_data =
      tflite::GetTensorData<int32_t>(segment_ids_tensor);

  const tflite::RuntimeShape output_shape =
      tflite::GetTensorShape(output_tensor);
  const int output_flat_size = output_shape.FlatSize();

  auto* const output_data = tflite::GetTensorData<float>(output_tensor);
  // The size of a single "row" in the output tensor. We use this to compute the
  // address of the segment in the output vector.
  const int row_size = tflite::FlatSizeSkipDim(output_shape, 0);
  std::fill(output_data, output_data + output_shape.FlatSize(), 0.0f);

  for (int segment_index = 0; segment_index < index_flat_size;
       ++segment_index) {
    const int segment = index_data[segment_index];
    const int input_row_start = segment_index * row_size;
    const int output_row_start = segment * row_size;
    TF_LITE_ENSURE(context, input_row_start + row_size <= data_flat_size);
    TF_LITE_ENSURE(context, output_row_start >= 0);
    // NOTE(ondrasej): This should also catch the case where the `num_segments`
    // input is smaller than the largest segment ID in `segment_ids`.
    TF_LITE_ENSURE(context, output_row_start + row_size <= output_flat_size);
    for (int i = 0; i < row_size; ++i) {
      output_data[output_row_start + i] += data[input_row_start + i];
    }
  }

  return kTfLiteOk;
}

}  // namespace

const char* kUnsortedSegmentSumOpName = "UnsortedSegmentSum";

TfLiteRegistration* RegisterUnsortedSegmentSumOp() {
  static TfLiteRegistration registration = {
      .init = nullptr,
      .free = nullptr,
      .prepare = UnsortedSegmentSumPrepare,
      .invoke = UnsortedSegmentSumInvoke,
  };
  return &registration;
}

}  // namespace gematria
