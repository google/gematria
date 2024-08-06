#!/bin/bash
#
# Copyright 2023 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Converts a frozen TensorFlow graph to a model in the .tflite format. To be
# processed with this script, the model must come from a Gematria model based on
# the TokenGraphBuilderModel class from
# gematria/granite/graph_builder_model_base.py with no additional tf.placeholder
# tensors.
#
# Typical use:
#   convert_gb_token_model_to_tflite.sh \
#     --gematria_input_graphdef /tmp/gnn_graph.pbtxt \
#     --gematria_output_tflite /tmp/gnn.tflite \
#     --gematria_export_as_seq2seq
#
# See g3doc/granite-inference-api.md for more details on exporting models to the
# .tflite format.

function print_error_and_exit() {
  echo "$1" > /dev/stderr
  exit 1
}

# Parse command-line flags.
# TODO(ondrasej): Consider using getopt instead of parsing the flags manually.
gematria_export_as_seq2seq=0
gematria_export_with_annotations=0
gematria_input_graphdef=""
gematria_output_tflite=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --gematria_input_graphdef)
      gematria_input_graphdef="$2"
      shift
      ;;
    --gematria_input_graphdef=*)
      gematria_input_graphdef="${1:26}"
      ;;
    --gematria_output_tflite)
      gematria_output_tflite="$2"
      shift
      ;;
    --gematria_output_tflite=*)
      gematria_output_tflite="${1:25}"
      ;;
    --gematria_export_as_seq2seq)
      gematria_export_as_seq2seq=1
      ;;
    --gematria_export_with_annotations)
      gematria_export_with_annotations=1
      ;;
    *)
      print_error_and_exit "Unexpected command-line argument: $1"
  esac
  shift
done

if [[ -z "${gematria_input_graphdef}" ]]; then
  print_error_and_exit "Flag --gematria_input_graphdef is missing."
fi
if [[ -z "${gematria_output_tflite}" ]]; then
  print_error_and_exit "Flag --gematria_output_tflite is missing."
fi

# Prints its arguments joined by a comma.
function str_join() {
  local IFS=","
  echo "$*"
}

# The list of inputs of the model. This must contain an entry for each
# tf.placeholder tensor used in the Python code.
declare -a INPUT_TENSORS_LIST=()
if (( gematria_export_as_seq2seq )); then
  INPUT_TENSORS_LIST+=( ModelBase.delta_block_index_tensor )
fi
INPUT_TENSORS_LIST+=(
  GnnModelBase.node_features
  GnnModelBase.edge_features
  GnnModelBase.global_features
  GnnModelBase.receivers
  GnnModelBase.senders
  GnnModelBase.num_edges
  GnnModelBase.num_nodes
)
if (( gematria_export_as_seq2seq ||
      gematria_export_with_annotations )); then
  INPUT_TENSORS_LIST+=( GraphBuilderModelBase.instruction_node_mask )
fi
if (( gematria_export_with_annotations )); then
  INPUT_TENSORS_LIST+=( GraphBuilderModelBase.instruction_annotations )
fi
readonly INPUT_TENSORS_LIST
INPUT_TENSORS=$(str_join "${INPUT_TENSORS_LIST[@]}")
readonly INPUT_TENSORS

# The list of TensorFlow ops used in the exported model.
readonly TARGET_OPS_LIST=(
  # The basic TensorFlow Lite ops.
  TFLITE_BUILTINS
)
TARGET_OPS=$(str_join "${TARGET_OPS_LIST[@]}")
readonly TARGET_OPS

declare -a OUTPUT_TENSORS_LIST=( ModelBase.output_tensor )
if (( gematria_export_as_seq2seq )); then
  OUTPUT_TENSORS_LIST+=( ModelBase.output_tensor_deltas )
fi
OUTPUT_TENSORS_LIST+=(
  TokenModel.token_list
  GraphBuilderModelBase.special_tokens
)
if (( gematria_export_with_annotations )); then
  OUTPUT_TENSORS_LIST+=( GraphBuilderModelBase.annotation_names )
fi
readonly OUTPUT_TENSORS_LIST
OUTPUT_TENSORS=$(str_join "${OUTPUT_TENSORS_LIST[@]}")
readonly OUTPUT_TENSORS

tflite_convert \
  --graph_def_file="${gematria_input_graphdef}" \
  --output_file="${gematria_output_tflite}" \
  --enable_v1_converter \
  --allow_custom_ops \
  --experimental_new_converter \
  --output_arrays="${OUTPUT_TENSORS}" \
  --input_arrays="${INPUT_TENSORS}" \
  --target_ops="${TARGET_OPS}"
