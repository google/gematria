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
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gematria/basic_block/basic_block.h"
#include "gematria/granite/graph_builder.h"
#include "gematria/model/oov_token_behavior.h"
#include "gematria/tflite/unsorted_segment_sum_op.h"
#include "gematria/utils/string.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"

namespace gematria {
namespace {

using ::tflite::FlatBufferModel;

// Graph builder models may have some valid combination of these input tensors
// depending on their configuration.
enum class InputTensor {
  kDeltaBlockIndex = 0,
  kGraphNodes,
  kGraphEdges,
  kGraphGlobals,
  kGraphReceivers,
  kGraphSenders,
  kGraphNEdge,
  kGraphNNode,
  kInstructionNodeMask,
};

// The names of the valid input tensors in the compiled TensorFlow Lite model.
// Not all tensors are present in all models.
constexpr std::string_view kDeltaBlockIndexTensorName =
    "ModelBase.delta_block_index_tensor";
constexpr std::string_view kGraphNodesTensorName = "GnnModelBase.node_features";
constexpr std::string_view kGraphEdgesTensorName = "GnnModelBase.edge_features";
constexpr std::string_view kGraphGlobalsTensorName =
    "GnnModelBase.global_features";
constexpr std::string_view kGraphReceiversTensorName = "GnnModelBase.receivers";
constexpr std::string_view kGraphSendersTensorName = "GnnModelBase.senders";
constexpr std::string_view kGraphNEdgeTensorName = "GnnModelBase.num_edges";
constexpr std::string_view kGraphNNodeTensorName = "GnnModelBase.num_nodes";
constexpr std::string_view kInstructionNodeMaskTensorName =
    "GraphBuilderModelBase.instruction_node_mask";

// Look-up table between `InputTensor` and input tensor names. Used by
// `GetInputTensorName` and `GetInputTensorFromName` to create an efficient
// mapping between tensor and their names at compile time.
constexpr std::pair<InputTensor, std::string_view> kPossibleInputTensors[]{
    {InputTensor::kDeltaBlockIndex, kDeltaBlockIndexTensorName},
    {InputTensor::kGraphNodes, kGraphNodesTensorName},
    {InputTensor::kGraphEdges, kGraphEdgesTensorName},
    {InputTensor::kGraphGlobals, kGraphGlobalsTensorName},
    {InputTensor::kGraphReceivers, kGraphReceiversTensorName},
    {InputTensor::kGraphSenders, kGraphSendersTensorName},
    {InputTensor::kGraphNEdge, kGraphNEdgeTensorName},
    {InputTensor::kGraphNNode, kGraphNNodeTensorName},
    {InputTensor::kInstructionNodeMask, kInstructionNodeMaskTensorName},
};

// Graph builder models can have up to these many different input tensors
// depending on their configuration, as enumerated by `InputTensor`.
constexpr int kNumPossibleInputTensors = std::size(kPossibleInputTensors);

// Basic checks to ensure the look-up table is consistent with `InputTensor`.
// The second assertion must be manually updated with the first `InputTensor`
// and the third assertion with the last `InputTensor` when a new one is added.
static_assert(static_cast<int>(kPossibleInputTensors[0].first) == 0,
              "The start of `kPossibleInputTensors` is not aligned with the "
              "start of `InputTensor`");
static_assert(static_cast<int>(InputTensor::kDeltaBlockIndex) == 0,
              "`InputTensor` does not begin enumerating from 0");
static_assert(
    static_cast<int>(InputTensor::kInstructionNodeMask) ==
        kNumPossibleInputTensors - 1,
    "`kPossibleInputTensors` does not have an entry for every `InputTensor`");

// Get the name associated with an input tensor.
constexpr std::string_view GetInputTensorName(const InputTensor tensor) {
  for (const auto& [other_tensor, name] : kPossibleInputTensors) {
    if (tensor == other_tensor) {
      return name;
    }
  }
}

// Get the input tensor associated with a name. `name` must be one of the
// input tensor names defined above, otherwise returns `std::nullopt`.
constexpr std::optional<InputTensor> GetInputTensorFromName(
    const std::string_view name) {
  for (const auto& [tensor, other_name] : kPossibleInputTensors) {
    if (name == other_name) {
      return tensor;
    }
  }
  return std::nullopt;
}

// The list of all input tensors that every graph builder model must have
// to be valid. Other tensors may be present based on the configuration.
constexpr InputTensor kRequiredInputTensors[]{
    InputTensor::kGraphNodes,   InputTensor::kGraphEdges,
    InputTensor::kGraphGlobals, InputTensor::kGraphReceivers,
    InputTensor::kGraphSenders, InputTensor::kGraphNEdge,
    InputTensor::kGraphNNode,   InputTensor::kInstructionNodeMask,
};

// The model must have at these many input tensors - it may not include some
// tensors such as the delta block index tensor depending on the model
// configuration.
constexpr int kNumRequiredInputTensors = std::size(kRequiredInputTensors);

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
constexpr std::string_view kNodeTokensTensorName = "TokenModel.token_list";
constexpr std::string_view kSpecialTokensTensorName =
    "GraphBuilderModelBase.special_tokens";

// Checks that:
// 1. `tensor` != nullptr,
// 2. `tensor` has type `tensor_type`.
// 3. `tensor` has the number of dimensions corresponding to the number of
// elements of `sizes`, and the sizes in those dimensions are equal to
// `sizes`. Returns `llvm::Error::success()` when all checks pass, an error
// otherwise.
//
// TODO(ondrasej): See if we can replace this function and the one below with
// TFModelEvaluatorImpl::checkReportAndInvalidate.
template <typename... Args>
llvm::Error CheckTensorTypeAndDimensions(int tensor_index,
                                         const TfLiteTensor* tensor,
                                         TfLiteType tensor_type,
                                         Args... sizes) {
  const int64_t sizes_array[] = {static_cast<int64_t>(sizes)...};
  const int num_dimensions = std::size(sizes_array);
  if (tensor == nullptr) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Input tensor was not found at index %d.",
                                   tensor_index);
  }
  if (tensor->type != tensor_type) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Input tensor at index %d has invalid type.",
                                   tensor_index);
  }
  if (tensor->dims->size != num_dimensions) {
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "Input tensor at index %d has invalid number of dimensions. Expected "
        "%d, found %d.",
        tensor_index, num_dimensions, tensor->dims->size);
  }
  for (int i = 0; i < num_dimensions; ++i) {
    if (tensor->dims->data[i] != sizes_array[i]) {
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "Input tensor at index %d has unexpected size at dimension %d",
          tensor_index, i);
    }
  }

  return llvm::Error::success();
}

// Similar to `CheckTensorTypeAndDimensions`, but tests `tensor->dims_signature`
// rather than `tensor->dims`.
template <typename... Args>
llvm::Error CheckTensorSignature(int tensor_index, const TfLiteTensor* tensor,
                                 Args... sizes) {
  const int sizes_array[] = {sizes...};
  const int num_dimensions = std::size(sizes_array);
  if (tensor == nullptr) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Input tensor was not found at index %d.",
                                   tensor_index);
  }
  if (tensor->dims_signature->size != num_dimensions) {
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "Input tensor at index %d has invalid number of dimensions. Expected "
        "%d, found %d.",
        tensor_index, num_dimensions, tensor->dims_signature->size);
  }
  for (int i = 0; i < num_dimensions; ++i) {
    if (tensor->dims_signature->data[i] != sizes_array[i]) {
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "Input tensor at index %d has unexpected size at dimension %d",
          tensor_index, i);
    }
  }

  return llvm::Error::success();
}

// Fills a 1D tensor at the given index in `interpreter` from the given
// std::vector. Returns an error if the tensor shape does not match the size of
// the vector or the type of the tensor does not match the requested tensor
// element type.
template <typename TensorElementType, typename InputElementType>
llvm::Error FillTensorFromStdVector(
    tflite::Interpreter* interpreter,
    const std::vector<InputElementType>& input_vector, int tensor_index) {
  const TfLiteTensor* const tensor = interpreter->input_tensor(tensor_index);
  if (llvm::Error error = CheckTensorTypeAndDimensions(
          tensor_index, tensor, tflite::typeToTfLiteType<TensorElementType>(),
          static_cast<int>(input_vector.size()))) {
    return error;
  }
  auto* const tensor_data =
      interpreter->typed_input_tensor<TensorElementType>(tensor_index);
  std::copy(input_vector.begin(), input_vector.end(), tensor_data);
  return llvm::Error::success();
}

// Fills a 2D tensor at the given index in `interpreter` from a given matrix
// represented as a vector of vectors. The data are added to the tensor in the
// natural ordering (the inner vectors of the matrix are concatenated into the
// flat tensor buffer).
// Returns an error if the tensor shape does not match the shape of the matrix
// or the type of the tensor does not match the requested tensor element type.
template <typename TensorElementType, typename InputElementType>
llvm::Error FillTensorFromStdVectorMatrix(
    tflite::Interpreter* interpreter,
    const std::vector<std::vector<InputElementType>>& input_matrix,
    int tensor_index) {
  const TfLiteTensor* const tensor = interpreter->input_tensor(tensor_index);
  if (llvm::Error error = CheckTensorTypeAndDimensions(
          tensor_index, tensor, tflite::typeToTfLiteType<TensorElementType>(),
          input_matrix.size(), input_matrix[0].size())) {
    return error;
  }

  const int expected_size = tensor->dims->data[1];
  auto* const tensor_data =
      interpreter->typed_input_tensor<TensorElementType>(tensor_index);
  for (int row = 0; row < input_matrix.size(); ++row) {
    const std::vector<TensorElementType>& row_data = input_matrix[row];
    if (expected_size != row_data.size()) {
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "Unexpected row data size. Expected %d, found %d", expected_size,
          row_data.size());
    }
    std::copy(row_data.begin(), row_data.end(),
              tensor_data + expected_size * row);
  }
  return llvm::Error::success();
}

// Resizes a 1D tensor at the given index in `interpreter` to a given size.
// Requires that the tensor is a 1D variable size tensor, i.e. its
// dims_signature is [-1].
llvm::Error Resize1DTensor(tflite::Interpreter* interpreter, int tensor_index,
                           int desired_size) {
  assert(interpreter != nullptr);
  const TfLiteTensor* const tensor = interpreter->input_tensor(tensor_index);
  if (llvm::Error error = CheckTensorSignature(tensor_index, tensor, -1)) {
    return error;
  }

  const TfLiteStatus status =
      interpreter->ResizeInputTensor(tensor_index, {desired_size});
  if (status != kTfLiteOk) {
    return llvm::make_error<llvm::StringError>(
        "Resizing the tensor failed with status " + llvm::Twine(status),
        llvm::errc::invalid_argument);
  }
  return llvm::Error::success();
}

// Resizes a 2D tensor at the given index in `interpreter` to a given batch
// size. Requires that the tensor is a 2D tensor where the first dimension has a
// variable size, i.e. its dims_signature is [-1, expected_second_dimension].
llvm::Error Resize2DTensor(tflite::Interpreter* interpreter, int tensor_index,
                           int desired_first_dimension_size,
                           int expected_second_dimension_size) {
  assert(interpreter != nullptr);
  const TfLiteTensor* const tensor = interpreter->input_tensor(tensor_index);
  if (llvm::Error error = CheckTensorSignature(
          tensor_index, tensor, -1, expected_second_dimension_size)) {
    return error;
  }
  const TfLiteStatus status = interpreter->ResizeInputTensor(
      tensor_index,
      {desired_first_dimension_size, expected_second_dimension_size});
  if (status != kTfLiteOk) {
    return llvm::make_error<llvm::StringError>(
        "Resizing the tensor failed with status " + llvm::Twine(status),
        llvm::errc::invalid_argument);
  }
  return llvm::Error::success();
}

// Finds a tensor in the model by its name. The name must be the same as used
// when constructing the graph. `tensor_indices` is the list of candidate tensor
// indices; typically, this is interpreter.inputs() or interpreter.outputs().
// Returns an error when the tensor is not found.
llvm::Expected<int> TensorIndexByName(const tflite::Interpreter& interpreter,
                                      llvm::ArrayRef<int> tensor_indices,
                                      std::string_view name) {
  for (const int tensor_index : tensor_indices) {
    const TfLiteTensor* const tensor = interpreter.tensor(tensor_index);
    if (name == tensor->name) {
      return tensor_index;
    }
  }
  return llvm::make_error<llvm::StringError>(
      "Tensor was not found " + llvm::Twine(name),
      llvm::errc::invalid_argument);
}

// Creates a new interpreter for `tflite_model`. Returns an error when the
// interpreter can't be created, e.g. when the model uses unsupported TensorFlow
// ops.
llvm::Expected<std::unique_ptr<tflite::Interpreter>> CreateInterpreter(
    const FlatBufferModel& tflite_model) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(kUnsortedSegmentSumOpName, RegisterUnsortedSegmentSumOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  const TfLiteStatus status =
      tflite::InterpreterBuilder(tflite_model, resolver)(&interpreter);
  if (status != kTfLiteOk) {
    return llvm::make_error<llvm::StringError>(
        "Could not create the interpreter.", llvm::errc::not_supported);
  }
  assert(interpreter != nullptr);
  return interpreter;
}

// Extracts the list of node tokens from the model. The token list should be a
// Const tensor, and as such, it should be readable without providing any
// inputs.
// Returns an error when the token list tensor is not found or when it is not
// readable.
llvm::Expected<std::vector<std::string>> GetNodeTokenList(
    const tflite::Interpreter& interpreter) {
  llvm::Expected<int> token_list_tensor_index = TensorIndexByName(
      interpreter, interpreter.outputs(), kNodeTokensTensorName);
  if (auto error = token_list_tensor_index.takeError()) return error;
  const TfLiteTensor* const token_list_tensor =
      interpreter.tensor(*token_list_tensor_index);
  assert(token_list_tensor != nullptr);

  const size_t token_list_size_bytes = token_list_tensor->bytes;
  // The token list tensor is a Const operation, so it should be readable before
  // running the inference or providing any inputs.
  const char* const token_list_raw_data = reinterpret_cast<const char*>(
      interpreter.typed_tensor<uint8_t>(*token_list_tensor_index));
  if (token_list_raw_data == nullptr) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "The token list could not be read");
  }
  const std::string_view token_list_data(token_list_raw_data,
                                         token_list_size_bytes);
  return StrSplitAsCopy(token_list_data, '\0');
}

// Returns token name from `node_token_list` at `token_index`. Returns an error
// when the token index is out of range.
llvm::Expected<std::string> GetNodeTokenAtIndex(
    const std::vector<std::string>& node_token_list, int token_index) {
  if (token_index < 0 || token_index >= node_token_list.size()) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "The token index is out of range: %d"
                                   ", node_token_list.size() is %d",
                                   token_index, node_token_list.size());
  }
  return node_token_list[token_index];
}

}  // namespace

llvm::Expected<std::unique_ptr<GraphBuilderModelInference>>
GraphBuilderModelInference::FromTfLiteModel(
    const tflite::FlatBufferModel* tflite_model) {
  if (tflite_model == nullptr) {
    return llvm::make_error<llvm::StringError>(
        "tflite_model must not be nullptr", llvm::errc::invalid_argument);
  }
  llvm::Expected<std::unique_ptr<tflite::Interpreter>> interpreter =
      CreateInterpreter(*tflite_model);
  if (llvm::Error error = interpreter.takeError()) return error;

  // Get a mapping between the input tensors used in the model and their
  // corresponding indices.
  std::vector<int> input_tensor_to_idx(kNumPossibleInputTensors, -1);
  std::vector<int> unexpected_input_indices;
  for (int idx : (*interpreter)->inputs()) {
    const std::string_view name = (*interpreter)->GetInputName(idx);
    std::optional<InputTensor> input_tensor = GetInputTensorFromName(name);

    // Store indices of any unexpected input tensors to return as an error.
    if (!input_tensor.has_value()) {
      unexpected_input_indices.push_back(idx);
    }
    input_tensor_to_idx[static_cast<int>(input_tensor.value())] = idx;
  }

  // Checks whether the model has all input tensors required to have for all
  // graph builder models.
  std::vector<InputTensor> missing_inputs;
  for (const InputTensor input : kRequiredInputTensors) {
    if (input_tensor_to_idx[static_cast<int>(input)] == -1)
      missing_inputs.push_back(input);
  }

  // Report any missing or unexpected extra input tensors and exit.
  if (!missing_inputs.empty() || !unexpected_input_indices.empty()) {
    std::stringstream buffer;
    if (!missing_inputs.empty()) {
      buffer << "Model is missing input tensors. ";
      for (const InputTensor missing_input : missing_inputs) {
        buffer << GetInputTensorName(missing_input) << ", ";
      }
      buffer.seekp(-2, std::ios_base::end);
      buffer << " were expected but not found.";
      if (!unexpected_input_indices.empty()) {
        buffer << '\n';
      }
    }
    if (!unexpected_input_indices.empty()) {
      buffer << "Model is has unexpected extra input tensors. ";
      for (const int idx : unexpected_input_indices) {
        buffer << (*interpreter)->GetInputName(idx) << " at index " << idx
               << ", ";
      }
      buffer.seekp(-2, std::ios_base::end);
      buffer << " were not expected but found.";
    }
    return llvm::createStringError(llvm::errc::invalid_argument, buffer.str());
  }

  // Infer the model configuration from the non-required input tensors present.
  const bool uses_deltas =
      input_tensor_to_idx[static_cast<int>(InputTensor::kDeltaBlockIndex)] !=
      -1;

  // Get the list of node tokens used in the model.
  llvm::Expected<std::vector<std::string>> node_token_list =
      GetNodeTokenList(**interpreter);
  if (llvm::Error error = node_token_list.takeError()) return error;

  // Get the values of the special tensors used in the model.
  llvm::Expected<int> special_tokens_tensor_index = TensorIndexByName(
      **interpreter, (*interpreter)->outputs(), kSpecialTokensTensorName);
  if (llvm::Error error = special_tokens_tensor_index.takeError()) {
    return error;
  }
  const TfLiteTensor* const special_tokens_tensor =
      (*interpreter)->tensor(*special_tokens_tensor_index);
  if (llvm::Error error = CheckTensorTypeAndDimensions(
          *special_tokens_tensor_index, special_tokens_tensor,
          tflite::typeToTfLiteType<int32_t>(), kNumSpecialNodeTokens)) {
    return error;
  }
  const int32_t* const special_tokens_tensor_data =
      (*interpreter)->typed_tensor<int32_t>(*special_tokens_tensor_index);
  if (special_tokens_tensor_data == nullptr) {
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "The special token index tensor could not be read");
  }

  // We'll be std::move()-ing the node list vector in the same function call
  // where we use the token names. To be safe from any move effects, we make a
  // copy of all the tokens instead of taking a const reference.
  llvm::Expected<std::string> immediate_token = GetNodeTokenAtIndex(
      *node_token_list, special_tokens_tensor_data[kSpecialNodeTokenImmediate]);
  if (llvm::Error error = immediate_token.takeError()) return error;

  llvm::Expected<std::string> fp_immediate_token = GetNodeTokenAtIndex(
      *node_token_list,
      special_tokens_tensor_data[kSpecialNodeTokenFpImmediate]);
  if (llvm::Error error = fp_immediate_token.takeError()) return error;
  llvm::Expected<std::string> address_token = GetNodeTokenAtIndex(
      *node_token_list, special_tokens_tensor_data[kSpecialNodeTokenAddress]);
  if (llvm::Error error = address_token.takeError()) return error;

  llvm::Expected<std::string> memory_token = GetNodeTokenAtIndex(
      *node_token_list, special_tokens_tensor_data[kSpecialNodeTokenMemory]);
  if (llvm::Error error = memory_token.takeError()) return error;
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
    llvm::Expected<std::string> replacement_token =
        GetNodeTokenAtIndex(*node_token_list, replacement_token_index);
    if (llvm::Error error = replacement_token.takeError()) return error;
    out_of_vocabulary_behavior = OutOfVocabularyTokenBehavior::ReplaceWithToken(
        *std::move(replacement_token));
  }

  auto graph_builder = std::make_unique<BasicBlockGraphBuilder>(
      *std::move(node_token_list), /* immediate_token = */ *immediate_token,
      /* fp_immediate_token = */ *fp_immediate_token,
      /* address_token = */ *address_token, /* memory_token = */ *memory_token,
      /* annotation_names = */ std::vector<std::string>(),
      /* out_of_vocabulary_behavior = */ out_of_vocabulary_behavior);

  // We can't use std::make_unique<GraphBuilderModelInference>(), because
  // std::make_unique<>() requires a public constructor.
  return std::unique_ptr<GraphBuilderModelInference>(
      new GraphBuilderModelInference(
          std::move(graph_builder), tflite_model, std::move(*interpreter),
          std::move(input_tensor_to_idx), uses_deltas));
}

GraphBuilderModelInference::GraphBuilderModelInference(
    std::unique_ptr<BasicBlockGraphBuilder> graph_builder,
    const FlatBufferModel* tflite_model,
    std::unique_ptr<tflite::Interpreter> interpreter,
    std::vector<int> input_tensor_to_idx, bool uses_deltas)
    : graph_builder_(std::move(graph_builder)),
      tflite_model_(*tflite_model),
      interpreter_(std::move(interpreter)),
      input_tensor_to_idx_(std::move(input_tensor_to_idx)),
      uses_deltas_(uses_deltas) {
  assert(tflite_model != nullptr);
  assert(interpreter_ != nullptr);
  assert(graph_builder_ != nullptr);
}

bool GraphBuilderModelInference::AddBasicBlockToBatch(const BasicBlock& block) {
  return graph_builder_->AddBasicBlock(block);
}

#define GEMATRIA_RETURN_IF_ERROR(statement)            \
  do {                                                 \
    if (llvm::Error error = (statement)) return error; \
  } while (false)

llvm::Expected<std::vector<GraphBuilderModelInference::OutputType>>
GraphBuilderModelInference::RunInference() {
  if (graph_builder_->num_graphs() == 0) {
    return std::vector<GraphBuilderModelInference::OutputType>();
  }

  const std::vector<bool> instruction_node_mask =
      graph_builder_->InstructionNodeMask();
  const std::vector<int> delta_block_index = graph_builder_->DeltaBlockIndex();

  // Resize the input tensors according to the size of the input data.
  GEMATRIA_RETURN_IF_ERROR(Resize1DTensor(
      interpreter_.get(),
      input_tensor_to_idx_[static_cast<int>(InputTensor::kGraphNodes)],
      graph_builder_->num_nodes()));
  GEMATRIA_RETURN_IF_ERROR(Resize1DTensor(
      interpreter_.get(),
      input_tensor_to_idx_[static_cast<int>(InputTensor::kGraphEdges)],
      graph_builder_->num_edges()));
  GEMATRIA_RETURN_IF_ERROR(Resize1DTensor(
      interpreter_.get(),
      input_tensor_to_idx_[static_cast<int>(InputTensor::kGraphReceivers)],
      static_cast<int>(graph_builder_->edge_receivers().size())));
  GEMATRIA_RETURN_IF_ERROR(Resize1DTensor(
      interpreter_.get(),
      input_tensor_to_idx_[static_cast<int>(InputTensor::kGraphSenders)],
      static_cast<int>(graph_builder_->edge_senders().size())));
  GEMATRIA_RETURN_IF_ERROR(Resize1DTensor(
      interpreter_.get(),
      input_tensor_to_idx_[static_cast<int>(InputTensor::kGraphNEdge)],
      static_cast<int>(graph_builder_->num_nodes_per_block().size())));
  GEMATRIA_RETURN_IF_ERROR(Resize1DTensor(
      interpreter_.get(),
      input_tensor_to_idx_[static_cast<int>(InputTensor::kGraphNNode)],
      static_cast<int>(graph_builder_->num_edges_per_block().size())));
  GEMATRIA_RETURN_IF_ERROR(Resize2DTensor(
      interpreter_.get(),
      input_tensor_to_idx_[static_cast<int>(InputTensor::kGraphGlobals)],
      /* desired_first_dimension_size = */ graph_builder_->num_graphs(),
      /* expected_second_dimension_size = */
      graph_builder_->num_node_tokens()));
  GEMATRIA_RETURN_IF_ERROR(Resize1DTensor(
      interpreter_.get(),
      input_tensor_to_idx_[static_cast<int>(InputTensor::kInstructionNodeMask)],
      static_cast<int>(instruction_node_mask.size())));
  if (uses_deltas_) {
    GEMATRIA_RETURN_IF_ERROR(Resize1DTensor(
        interpreter_.get(),
        input_tensor_to_idx_[static_cast<int>(InputTensor::kDeltaBlockIndex)],
        static_cast<int>(delta_block_index.size())));
  }

  if (const TfLiteStatus status = interpreter_->AllocateTensors();
      status != kTfLiteOk) {
    return llvm::make_error<llvm::StringError>(
        "Could not allocate memory for tensors", llvm::errc::not_enough_memory);
  }

  // Fill in the input tensors.
  GEMATRIA_RETURN_IF_ERROR(FillTensorFromStdVector<int32_t>(
      interpreter_.get(), graph_builder_->node_features(),
      input_tensor_to_idx_[static_cast<int>(InputTensor::kGraphNodes)]));
  GEMATRIA_RETURN_IF_ERROR(FillTensorFromStdVector<int32_t>(
      interpreter_.get(), graph_builder_->EdgeFeatures(),
      input_tensor_to_idx_[static_cast<int>(InputTensor::kGraphEdges)]));
  GEMATRIA_RETURN_IF_ERROR(FillTensorFromStdVector<int32_t>(
      interpreter_.get(), graph_builder_->edge_receivers(),
      input_tensor_to_idx_[static_cast<int>(InputTensor::kGraphReceivers)]));
  GEMATRIA_RETURN_IF_ERROR(FillTensorFromStdVector<int32_t>(
      interpreter_.get(), graph_builder_->edge_senders(),
      input_tensor_to_idx_[static_cast<int>(InputTensor::kGraphSenders)]));
  GEMATRIA_RETURN_IF_ERROR(FillTensorFromStdVector<int32_t>(
      interpreter_.get(), graph_builder_->num_nodes_per_block(),
      input_tensor_to_idx_[static_cast<int>(InputTensor::kGraphNNode)]));
  GEMATRIA_RETURN_IF_ERROR(FillTensorFromStdVector<int32_t>(
      interpreter_.get(), graph_builder_->num_edges_per_block(),
      input_tensor_to_idx_[static_cast<int>(InputTensor::kGraphNEdge)]));
  GEMATRIA_RETURN_IF_ERROR(FillTensorFromStdVectorMatrix<int32_t>(
      interpreter_.get(), graph_builder_->global_features(),
      input_tensor_to_idx_[static_cast<int>(InputTensor::kGraphGlobals)]));
  GEMATRIA_RETURN_IF_ERROR(
      FillTensorFromStdVector<bool>(interpreter_.get(), instruction_node_mask,
                                    input_tensor_to_idx_[static_cast<int>(
                                        InputTensor::kInstructionNodeMask)]));
  if (uses_deltas_) {
    GEMATRIA_RETURN_IF_ERROR(FillTensorFromStdVector<int32_t>(
        interpreter_.get(), delta_block_index,
        input_tensor_to_idx_[static_cast<int>(InputTensor::kDeltaBlockIndex)]));
  }

  if (const TfLiteStatus status = interpreter_->Invoke(); status != kTfLiteOk) {
    return llvm::make_error<llvm::StringError>(
        "Invoking the TensorFlow Lite interpreter failed",
        llvm::errc::io_error);
  }

  const TfLiteTensor* const output_tensor = interpreter_->output_tensor(0);
  if (output_tensor == nullptr) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "No output tensor at index 0.");
  }
  if (output_tensor->dims->size != 2) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Unexpected number of dimensions of the "
                                   "output tensor. Expected 2, found %d",
                                   output_tensor->dims->size);
  }
  if (output_tensor->dims->data[0] != graph_builder_->num_graphs()) {
    return llvm::createStringError(llvm::errc::result_out_of_range,
                                   "Unexpected number of rows in the output "
                                   "tensor. Expected %d, found %d.",
                                   graph_builder_->num_graphs(),
                                   output_tensor->dims->data[0]);
  }
  const int num_tasks = output_tensor->dims->data[1];
  auto* const output_tensor_data = interpreter_->typed_output_tensor<float>(0);
  assert(output_tensor_data != nullptr);

  std::vector<OutputType> output;
  output.reserve(graph_builder_->num_graphs());
  for (int i = 0; i < graph_builder_->num_graphs(); ++i) {
    output.emplace_back(output_tensor_data + i * num_tasks,
                        output_tensor_data + (i + 1) * num_tasks);
  }
  return output;
}

#undef GEMATRIA_RETURN_IF_ERROR

void GraphBuilderModelInference::Reset() { graph_builder_->Reset(); }

}  // namespace gematria
