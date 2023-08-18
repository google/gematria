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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gematria/basic_block/basic_block.h"
#include "gematria/granite/graph_builder.h"
#include "gematria/model/oov_token_behavior.h"
#include "gematria/tflite/unsorted_segment_sum_op.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace gematria {
namespace {

using ::tflite::FlatBufferModel;

// The indices of the input tensors in the compiled TensorFlow Lite model. This
// order of the input tensors must be preserved during the conversion of the
// model to the .tflite format.
constexpr int kDeltaBlockIndexTensor = 0;
constexpr int kGraphNodesTensor = 1;
constexpr int kGraphEdgesTensor = 2;
constexpr int kGraphGlobalsTensor = 3;
constexpr int kGraphReceiversTensor = 4;
constexpr int kGraphSendersTensor = 5;
constexpr int kGraphNEdgeTensor = 6;
constexpr int kGraphNNodeTensor = 7;
constexpr int kInstructionNodeMaskTensor = 8;

constexpr int kNumInputTensors = 9;

// The indices of special node token indices in the tensor
// `GraphBuilderModelBase.special_tokens`. For example the token used for
// address computation nodes can be obtained as
// node_token_list[special_node_tokens[kSpecialNodeTokenAddress]], where
// node_token_list is a string list extracted from the byte tensor
// `TokenModel.token_list` and `special_node_tokens` is the int32 tensor
// `GraphBuilderModelBase.special_tokens`.
constexpr int kSpecialNodeTokenImmediate = 0;
constexpr int kSpecialNodeTokenFpImmediate = 1;
constexpr int kSpecialNodeTokenAddress = 2;
constexpr int kSpecialNodeTokenMemory = 3;
constexpr int kSpecialNodeTokenReplacement = 4;

// The number of special node tokens; this is also the expected size of the
// special node token tensor.
constexpr int kNumSpecialNodeTokens = 5;

// The names of the tensors that contain the configuration of the graph builder
// class.
constexpr absl::string_view kNodeTokensTensorName = "TokenModel.token_list";
constexpr absl::string_view kSpecialTokensTensorName =
    "GraphBuilderModelBase.special_tokens";

// Fills a 1D tensor at the given index in `interpreter` from the given
// std::vector. CHECK-fails if the tensor shape does not match the size of the
// vector or the type of the tensor does not match the requested tensor element
// type.
template <typename TensorElementType, typename InputElementType>
void FillTensorFromStdVector(tflite::Interpreter* interpreter,
                             const std::vector<InputElementType>& input_vector,
                             int tensor_index) {
  const TfLiteTensor* const tensor = interpreter->input_tensor(tensor_index);
  ABSL_CHECK_EQ(tensor->type, tflite::typeToTfLiteType<TensorElementType>());
  ABSL_CHECK_EQ(tensor->dims->size, 1);
  ABSL_CHECK_EQ(tensor->dims->data[0], input_vector.size());

  auto* const tensor_data =
      interpreter->typed_input_tensor<TensorElementType>(tensor_index);
  std::copy(input_vector.begin(), input_vector.end(), tensor_data);
}

// Fills a 2D tensor at the given index in `interpreter` from a given matrix
// represented as a vector of vectors. The data are added to the tensor in the
// natural ordering (the inner vectors of the matrix are concatenated into the
// flat tensor buffer).
// CHECK-fails if the tensor shape does not match the shape of the matrix or the
// type of the tensor does not match the requested tensor element type.
template <typename TensorElementType, typename InputElementType>
void FillTensorFromStdVectorMatrix(
    tflite::Interpreter* interpreter,
    const std::vector<std::vector<InputElementType>>& input_matrix,
    int tensor_index) {
  const TfLiteTensor* const tensor = interpreter->input_tensor(tensor_index);
  ABSL_CHECK_EQ(tensor->type, tflite::typeToTfLiteType<TensorElementType>());
  ABSL_CHECK_EQ(tensor->dims->size, 2);
  ABSL_CHECK_EQ(tensor->dims->data[0], input_matrix.size());

  const int expected_size = tensor->dims->data[1];
  auto* const tensor_data =
      interpreter->typed_input_tensor<TensorElementType>(tensor_index);
  for (int row = 0; row < input_matrix.size(); ++row) {
    const std::vector<TensorElementType>& row_data = input_matrix[row];
    ABSL_CHECK_EQ(expected_size, row_data.size());
    std::copy(row_data.begin(), row_data.end(),
              tensor_data + expected_size * row);
  }
}

// Resizes a 1D tensor at the given index in `interpreter` to a given size.
// Requires that the tensor is a 1D variable size tensor, i.e. its
// dims_signature is [-1].
void Resize1DTensor(tflite::Interpreter* interpreter, int tensor_index,
                    int desired_size) {
  ABSL_CHECK(interpreter != nullptr);
  const TfLiteTensor* const tensor = interpreter->input_tensor(tensor_index);
  ABSL_CHECK(tensor != nullptr);
  ABSL_CHECK(tensor->dims_signature != nullptr);
  ABSL_CHECK_EQ(tensor->dims_signature->size, 1);
  ABSL_CHECK_EQ(tensor->dims_signature->data[0], -1);

  const TfLiteStatus status =
      interpreter->ResizeInputTensor(tensor_index, {desired_size});
  ABSL_CHECK_EQ(status, kTfLiteOk);
}

// Resizes a 2D tensor at the given index in `interpreter` to a given batch
// size. Requires that the tensor is a 2D tensor where the first dimension has a
// variable size, i.e. its dims_signature is [-1, expected_second_dimension].
void Resize2DTensor(tflite::Interpreter* interpreter, int tensor_index,
                    int desired_first_dimension_size,
                    int expected_second_dimension_size) {
  ABSL_CHECK(interpreter != nullptr);
  const TfLiteTensor* const tensor = interpreter->input_tensor(tensor_index);
  ABSL_CHECK(tensor != nullptr);
  ABSL_CHECK(tensor->dims_signature != nullptr);
  ABSL_CHECK_EQ(tensor->dims_signature->size, 2);
  ABSL_CHECK_EQ(tensor->dims_signature->data[0], -1);
  ABSL_CHECK_EQ(tensor->dims_signature->data[1],
                expected_second_dimension_size);

  const TfLiteStatus status = interpreter->ResizeInputTensor(
      tensor_index,
      {desired_first_dimension_size, expected_second_dimension_size});
  ABSL_CHECK_EQ(status, kTfLiteOk);
}

// Finds a tensor in the model by its name. The name must be the same as used
// when constructing the graph. `tensor_indices` is the list of candidate tensor
// indices; typically, this is interpreter.inputs() or interpreter.outputs().
// Returns an error when the tensor is not found.
absl::StatusOr<int> TensorIndexByName(const tflite::Interpreter& interpreter,
                                      absl::Span<const int> tensor_indices,
                                      absl::string_view name) {
  for (const int tensor_index : tensor_indices) {
    const TfLiteTensor* const tensor = interpreter.tensor(tensor_index);
    if (name == tensor->name) {
      return tensor_index;
    }
  }
  return absl::NotFoundError(absl::StrCat("Tensor was not found: ", name));
}

// Creates a new interpreter for `tflite_model`. Returns an error when the
// interpreter can't be created, e.g. when the model uses unsupported TensorFlow
// ops.
absl::StatusOr<std::unique_ptr<tflite::Interpreter>> CreateInterpreter(
    const FlatBufferModel& tflite_model) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(kUnsortedSegmentSumOpName, RegisterUnsortedSegmentSumOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  const TfLiteStatus status =
      tflite::InterpreterBuilder(tflite_model, resolver)(&interpreter);
  if (status != kTfLiteOk) {
    return absl::FailedPreconditionError("Could not create the interpreter.");
  }
  ABSL_CHECK(interpreter != nullptr);

  return interpreter;
}

// Extracts the list of node tokens from the model. The token list should be a
// Const tensor, and as such, it should be readable without providing any
// inputs.
// Returns an error when the token list tensor is not found or when it is not
// readable.
absl::StatusOr<std::vector<std::string>> GetNodeTokenList(
    const tflite::Interpreter& interpreter) {
  const absl::StatusOr<int> token_list_tensor_index = TensorIndexByName(
      interpreter, interpreter.outputs(), kNodeTokensTensorName);
  if (!token_list_tensor_index.ok()) return token_list_tensor_index.status();
  const TfLiteTensor* const token_list_tensor =
      interpreter.tensor(*token_list_tensor_index);
  ABSL_CHECK(token_list_tensor != nullptr);

  const size_t token_list_size_bytes = token_list_tensor->bytes;
  // The token list tensor is a Const operation, so it should be readable before
  // running the inference or providing any inputs.
  const char* const token_list_raw_data = reinterpret_cast<const char*>(
      interpreter.typed_tensor<uint8_t>(*token_list_tensor_index));
  if (token_list_raw_data == nullptr) {
    return absl::FailedPreconditionError("The token list could not be read");
  }
  const absl::string_view token_list_data(token_list_raw_data,
                                          token_list_size_bytes);
  return absl::StrSplit(token_list_data, '\0');
}

// Returns token name from `node_token_list` at `token_index`. Returns an error
// when the token index is out of range.
absl::StatusOr<std::string> GetNodeTokenAtIndex(
    const std::vector<std::string>& node_token_list, int token_index) {
  if (token_index < 0 || token_index >= node_token_list.size()) {
    return absl::FailedPreconditionError(
        absl::StrCat("The token index is out of range: ", token_index,
                     ",node_token_list.size() is ", node_token_list.size()));
  }
  return node_token_list[token_index];
}

}  // namespace

absl::StatusOr<std::unique_ptr<GraphBuilderModelInference>>
GraphBuilderModelInference::FromTfLiteModel(
    const tflite::FlatBufferModel* tflite_model) {
  if (tflite_model == nullptr) {
    return absl::InvalidArgumentError("tflite_model must not be nullptr");
  }
  const absl::StatusOr<std::unique_ptr<tflite::Interpreter>> interpreter =
      CreateInterpreter(*tflite_model);
  if (!interpreter.ok()) return interpreter.status();

  // Get the list of node tokens used in the model.
  absl::StatusOr<std::vector<std::string>> node_token_list =
      GetNodeTokenList(**interpreter);
  if (!node_token_list.ok()) return node_token_list.status();

  // Get the values of the special tensors used in the model.
  const absl::StatusOr<int> special_tokens_tensor_index = TensorIndexByName(
      **interpreter, (*interpreter)->outputs(), kSpecialTokensTensorName);
  if (!special_tokens_tensor_index.ok()) {
    return special_tokens_tensor_index.status();
  }
  const TfLiteTensor* const special_tokens_tensor =
      (*interpreter)->tensor(*special_tokens_tensor_index);
  ABSL_CHECK(special_tokens_tensor != nullptr);
  ABSL_CHECK(special_tokens_tensor->dims != nullptr);
  if (special_tokens_tensor->dims->size != 1 ||
      special_tokens_tensor->dims->data[0] != kNumSpecialNodeTokens) {
    return absl::FailedPreconditionError(
        "The special node token tensor has an unexpected structure");
  }
  const int32_t* const special_tokens_tensor_data =
      (*interpreter)->typed_tensor<int32_t>(*special_tokens_tensor_index);
  if (special_tokens_tensor_data == nullptr) {
    return absl::FailedPreconditionError(
        "The special token index tensor could not be read");
  }
  // We'll be std::move()-ing the node list vector in the same function call
  // where we use the token names. To be safe from any move effects, we make a
  // copy of all the tokens instead of taking a const reference.
  const absl::StatusOr<std::string> immediate_token = GetNodeTokenAtIndex(
      *node_token_list, special_tokens_tensor_data[kSpecialNodeTokenImmediate]);
  if (!immediate_token.ok()) return immediate_token.status();

  const absl::StatusOr<std::string> fp_immediate_token = GetNodeTokenAtIndex(
      *node_token_list,
      special_tokens_tensor_data[kSpecialNodeTokenFpImmediate]);
  if (!fp_immediate_token.ok()) return fp_immediate_token.status();
  const absl::StatusOr<std::string> address_token = GetNodeTokenAtIndex(
      *node_token_list, special_tokens_tensor_data[kSpecialNodeTokenAddress]);
  if (!address_token.ok()) return address_token.status();

  const absl::StatusOr<std::string> memory_token = GetNodeTokenAtIndex(
      *node_token_list, special_tokens_tensor_data[kSpecialNodeTokenMemory]);
  if (!memory_token.ok()) return memory_token.status();
  const int32_t replacement_token_index =
      special_tokens_tensor_data[kSpecialNodeTokenReplacement];
  // The out-of-vocabulary behavior is represented implicitly in the tensor:
  //   - when a replacement token is specified, the behavior is to replace
  //     unknown tokens with this token.
  //   - when it is not specified (the index is < 0), the behavior is to return
  //     an error on unknown tokens.
  OutOfVocabularyTokenBehavior out_of_vocabulary_behavior =
      OutOfVocabularyTokenBehavior::ReturnError();
  if (replacement_token_index >= 0) {
    absl::StatusOr<std::string> replacement_token =
        GetNodeTokenAtIndex(*node_token_list, replacement_token_index);
    if (!replacement_token.ok()) return replacement_token.status();
    out_of_vocabulary_behavior = OutOfVocabularyTokenBehavior::ReplaceWithToken(
        *std::move(replacement_token));
  }

  auto graph_builder = std::make_unique<BasicBlockGraphBuilder>(
      *std::move(node_token_list), /* immediate_token = */ *immediate_token,
      /* fp_immediate_token = */ *fp_immediate_token,
      /* address_token = */ *address_token, /* memory_token = */ *memory_token,
      /* out_of_vocabulary_behavior = */ out_of_vocabulary_behavior);

  // We can't use std::make_unique<GraphBuilderModelInference>(), because
  // std::make_unique<>() requires a public constructor.
  return std::unique_ptr<GraphBuilderModelInference>(
      new GraphBuilderModelInference(std::move(graph_builder), tflite_model));
}

GraphBuilderModelInference::GraphBuilderModelInference(
    std::unique_ptr<BasicBlockGraphBuilder> graph_builder,
    const FlatBufferModel* tflite_model)
    : graph_builder_(std::move(graph_builder)),
      tflite_model_(*ABSL_DIE_IF_NULL(tflite_model)) {
  ABSL_CHECK(graph_builder_ != nullptr);
}

bool GraphBuilderModelInference::AddBasicBlockToBatch(const BasicBlock& block) {
  return graph_builder_->AddBasicBlock(block);
}

absl::StatusOr<std::vector<GraphBuilderModelInference::OutputType>>
GraphBuilderModelInference::RunInference() {
  if (graph_builder_->num_graphs() == 0) {
    return std::vector<GraphBuilderModelInference::OutputType>();
  }

  // TODO(ondrasej): Reuse the interpreter across RunInference() calls. The
  // graph builder class is already stateful, so this should not be an issue
  // and it could save us some loading time.
  absl::StatusOr<std::unique_ptr<tflite::Interpreter>> interpreter =
      CreateInterpreter(tflite_model_);
  if (!interpreter.ok()) return interpreter.status();

  // TODO(ondrasej): Move all the checks of the model format to the
  // initialization of the class. Ideally, return an absl::Status rather than
  // CHECK-fail when the model does not meet expectations.
  ABSL_CHECK_EQ((*interpreter)->inputs().size(), kNumInputTensors);

  const std::vector<bool> instruction_node_mask =
      graph_builder_->InstructionNodeMask();
  const std::vector<int> delta_block_index = graph_builder_->DeltaBlockIndex();

  // Resize the input tensors according to the size of the input data.
  // TODO(ondrasej): Replace the index-based lookups with name-based lookups.
  Resize1DTensor(interpreter->get(), kDeltaBlockIndexTensor,
                 static_cast<int>(delta_block_index.size()));
  Resize1DTensor(interpreter->get(), kGraphNodesTensor,
                 graph_builder_->num_nodes());
  Resize1DTensor(interpreter->get(), kGraphEdgesTensor,
                 graph_builder_->num_edges());
  Resize1DTensor(interpreter->get(), kGraphReceiversTensor,
                 static_cast<int>(graph_builder_->edge_receivers().size()));
  Resize1DTensor(interpreter->get(), kGraphSendersTensor,
                 static_cast<int>(graph_builder_->edge_senders().size()));
  Resize1DTensor(
      interpreter->get(), kGraphNEdgeTensor,
      static_cast<int>(graph_builder_->num_nodes_per_block().size()));
  Resize1DTensor(
      interpreter->get(), kGraphNNodeTensor,
      static_cast<int>(graph_builder_->num_edges_per_block().size()));
  Resize1DTensor(interpreter->get(), kInstructionNodeMaskTensor,
                 static_cast<int>(instruction_node_mask.size()));
  Resize2DTensor(
      interpreter->get(), kGraphGlobalsTensor,
      /* desired_first_dimension_size = */ graph_builder_->num_graphs(),
      /* expected_second_dimension_size = */ graph_builder_->num_node_tokens());

  if (const TfLiteStatus status = (*interpreter)->AllocateTensors();
      status != kTfLiteOk) {
    return absl::UnknownError("Could not allocate memory for tensors");
  }

  // Fill in the input tensors.
  FillTensorFromStdVector<int32_t>(interpreter->get(), delta_block_index,
                                   kDeltaBlockIndexTensor);
  FillTensorFromStdVector<int32_t>(
      interpreter->get(), graph_builder_->node_features(), kGraphNodesTensor);
  FillTensorFromStdVector<int32_t>(
      interpreter->get(), graph_builder_->EdgeFeatures(), kGraphEdgesTensor);
  FillTensorFromStdVector<int32_t>(interpreter->get(),
                                   graph_builder_->edge_receivers(),
                                   kGraphReceiversTensor);
  FillTensorFromStdVector<int32_t>(
      interpreter->get(), graph_builder_->edge_senders(), kGraphSendersTensor);
  FillTensorFromStdVector<int32_t>(interpreter->get(),
                                   graph_builder_->num_nodes_per_block(),
                                   kGraphNNodeTensor);
  FillTensorFromStdVector<int32_t>(interpreter->get(),
                                   graph_builder_->num_edges_per_block(),
                                   kGraphNEdgeTensor);
  FillTensorFromStdVector<bool>(interpreter->get(), instruction_node_mask,
                                kInstructionNodeMaskTensor);
  FillTensorFromStdVectorMatrix<int32_t>(interpreter->get(),
                                         graph_builder_->global_features(),
                                         kGraphGlobalsTensor);

  if (const TfLiteStatus status = (*interpreter)->Invoke();
      status != kTfLiteOk) {
    return absl::UnknownError(
        "Invoking the TensorFlow Lite interpreter failed");
  }

  const TfLiteTensor* const output_tensor = (*interpreter)->output_tensor(0);
  ABSL_CHECK(output_tensor != nullptr);
  ABSL_CHECK_EQ(output_tensor->dims->size, 2);
  ABSL_CHECK_EQ(output_tensor->dims->data[0], graph_builder_->num_graphs());
  const int num_tasks = output_tensor->dims->data[1];
  auto* const output_tensor_data =
      (*interpreter)->typed_output_tensor<float>(0);
  ABSL_CHECK(output_tensor_data != nullptr);

  std::vector<absl::InlinedVector<float, 4>> output;
  output.reserve(graph_builder_->num_graphs());
  for (int i = 0; i < graph_builder_->num_graphs(); ++i) {
    output.emplace_back(output_tensor_data + i * num_tasks,
                        output_tensor_data + (i + 1) * num_tasks);
  }
  return output;
}

void GraphBuilderModelInference::Reset() { graph_builder_->Reset(); }

}  // namespace gematria
