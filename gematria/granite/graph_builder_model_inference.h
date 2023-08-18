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

#ifndef THIRD_PARTY_GEMATRIA_GEMATRIA_GRANITE_GRAPH_BUILDER_MODEL_INFERENCE_H_
#define THIRD_PARTY_GEMATRIA_GEMATRIA_GRANITE_GRAPH_BUILDER_MODEL_INFERENCE_H_

#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "gematria/basic_block/basic_block.h"
#include "gematria/granite/graph_builder.h"
#include "tensorflow/lite/model_builder.h"

namespace gematria {

// Runs inference with a trained GRANITE model. The class uses TensorFlow Lite
// and a model stored in the .tflite format to do the inference in-process.
//
// This class works with all models based on the
// graph_builder_model_base.GraphBuilderModelBase unless the model adds more
// placeholder tensors to the graph.
//
// Typical usage:
//   auto tflite_model = tflite::FlatBufferModel::BuildFromFile(...);
//   absl::StatusOr<std::unique_ptr<GraphBuilderModelInference>>
//       inference_or_status =
//       GraphBuilderModelInference::FromTfLiteModel(tflite_model.get());
//   for (const BasicBlock& block : input_basic_blocks) {
//     inference.AddBasicBlockToBatch(block);
//   }
//   const auto predictions = inference.RunInference();
class GraphBuilderModelInference {
 public:
  // The type used for predictions for a single basic block. All Gematria models
  // are treated as multi-task models (even if they support just a single task)
  // and the number of tasks is a property of the model.
  // During inference, we return the outputs for all tasks (the values from all
  // output heads of the model) in a vector-like data structure. The precise
  // definition of this type may change in the future.
  using OutputType = absl::InlinedVector<float, 4>;

  // Creates the inference object from a model stored in the .tflite format.
  // Expects that the .tflite model contains also the definitions of node tokens
  // and creates a graph builder internally based on these definitions. Returns
  // an error when the model can't be loaded or it does not have all components
  // including the node token definitions.
  // Does not take ownership of `tflite_model`; the object must remain alive for
  // the whole lifetime of the inference object.
  static absl::StatusOr<std::unique_ptr<GraphBuilderModelInference>>
  FromTfLiteModel(const tflite::FlatBufferModel* tflite_model);

  // Adds a basic block to the current batch. Returns true when the basic block
  // was successfully added, otherwise false.
  // TODO(ondrasej): Add API that would allow rejecting blocks with unknown
  // tokens even if a replacement token was specified.
  bool AddBasicBlockToBatch(const BasicBlock& block);

  // Runs inference on the current batch. Returns a vector that contains
  // predictions for all basic blocks from the current batch in the order in
  // which they are added. The output for each basic block are the predictions
  // from all heads of the model.
  absl::StatusOr<std::vector<OutputType>> RunInference();

  // Removes all basic blocks from the current batch. Note that RunInference()
  // does not call this method automatically.
  // TODO(ondrasej): See if this method could be removed from the API.
  void Reset();

 private:
  // Creates the inference object for the given graph builder object and the
  // given model in the .tflite format. Note that `graph_builder` is a property
  // of the trained model and it should be set up the same way as during the
  // training of the model.
  // Causes a CHECK-failure if the model inputs and outputs do not match the
  // structure of a model based on the BasicBlockGraphBuilder class.
  GraphBuilderModelInference(
      std::unique_ptr<BasicBlockGraphBuilder> graph_builder,
      const tflite::FlatBufferModel* tflite_model);

  std::unique_ptr<BasicBlockGraphBuilder> graph_builder_;
  const tflite::FlatBufferModel& tflite_model_;
};

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_GRANITE_GRAPH_BUILDER_MODEL_INFERENCE_H_
