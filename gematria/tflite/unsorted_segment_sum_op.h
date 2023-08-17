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

// Implements `tf.UnsortedSegmentSum` as a custom TensorFlow Lite op. This
// op supports the same shapes of inputs and outputs as the original TensorFlow
// op, but it supports only float32 as data type, and int32 as index type.

#ifndef THIRD_PARTY_GEMATRIA_GEMATRIA_TFLITE_UNSORTED_SEGMENT_SUM_OP_H_
#define THIRD_PARTY_GEMATRIA_GEMATRIA_TFLITE_UNSORTED_SEGMENT_SUM_OP_H_

#include "tensorflow/lite/c/common.h"

namespace gematria {

// The name of the unsorted segment sum op that matches the name used by the
// TensorFlow Lite converter tool.
extern const char* kUnsortedSegmentSumOpName;

// Creates a registration structure for the unsorted segment sum op.
TfLiteRegistration* RegisterUnsortedSegmentSumOp();

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_TFLITE_UNSORTED_SEGMENT_SUM_OP_H_
