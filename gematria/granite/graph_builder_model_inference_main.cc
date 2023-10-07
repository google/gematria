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

// Runs Gematria model inference locally on basic blocks from a file.
//
// Typical usage:
//   graph_builder_model_inference_main \
//     --gematria_tflite_file models/granite_model.tflite \
//     --gematria_basic_block_hex_file /dev/stdin

#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "base/init_google.h"
#include "gematria/basic_block/basic_block.h"
#include "gematria/granite/graph_builder_model_inference.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/llvm/llvm_to_absl.h"
#include "gematria/utils/string.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Support/Error.h"
#include "tensorflow/lite/model_builder.h"

ABSL_FLAG(std::string, gematria_tflite_file, "",
          "The path to the .tflite file that contains the trained model.");
ABSL_FLAG(std::string, gematria_basic_block_hex_file, "",
          "The file from which the tool reads basic blocks in the hex format "
          "used in the BHive data set, one basic block per line.");
ABSL_FLAG(int, gematria_max_blocks_per_batch, std::numeric_limits<int>::max(),
          "The maximal number of blocks per batch. When non-positive, all "
          "blocks are put into the same batch.");

namespace gematria {
namespace {

void PrintPredictionsToStdout(
    const GraphBuilderModelInference::OutputType& predictions) {
  for (int i = 0; i < predictions.size(); ++i) {
    if (i > 0) std::cout << ",";
    std::cout << predictions[i];
  }
}

absl::Status ProcessBasicBlocksFromCommandLineFlags() {
  constexpr char kLlvmTriple[] = "x86_64-unknown-unknown";
  absl::StatusOr<std::unique_ptr<LlvmArchitectureSupport>> llvm_support =
      LlvmExpectedToStatusOr(
          LlvmArchitectureSupport::FromTriple(kLlvmTriple, "", ""));
  if (!llvm_support.ok()) return llvm_support.status();

  const std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(
          absl::GetFlag(FLAGS_gematria_tflite_file).c_str());
  if (model == nullptr) {
    return absl::UnknownError("Could not load the ");
  }
  absl::StatusOr<std::unique_ptr<GraphBuilderModelInference>>
      inference_or_status =
          GraphBuilderModelInference::FromTfLiteModel(model.get());
  if (!inference_or_status.ok()) return inference_or_status.status();
  GraphBuilderModelInference& inference = **inference_or_status;

  X86Canonicalizer canonicalizer(&(*llvm_support)->target_machine());

  std::unique_ptr<llvm::MCInstPrinter> inst_printer =
      (*llvm_support)->CreateMCInstPrinter(0);

  const auto run_inference_for_batch =
      [&](const std::vector<BasicBlock>& batch) -> absl::Status {
    inference.Reset();
    std::vector<bool> is_valid_block;
    is_valid_block.reserve(batch.size());
    for (const BasicBlock& block : batch) {
      is_valid_block.push_back(inference.AddBasicBlockToBatch(block));
      if (!is_valid_block.back()) {
        std::cerr << "Invalid basic block:\n" << block.ToString() << "\n";
      }
    }
    const absl::StatusOr<std::vector<GraphBuilderModelInference::OutputType>>
        predictions = inference.RunInference();
    if (!predictions.ok()) return predictions.status();

    int prediction_index = 0;
    for (const bool is_valid : is_valid_block) {
      if (is_valid) {
        PrintPredictionsToStdout((*predictions)[prediction_index++]);
      } else {
        std::cout << "Invalid block";
      }
      std::cout << std::endl;
    }
    return absl::OkStatus();
  };

  const int max_num_blocks_per_batch =
      absl::GetFlag(FLAGS_gematria_max_blocks_per_batch);
  const std::string hex_file_name =
      absl::GetFlag(FLAGS_gematria_basic_block_hex_file);
  std::vector<BasicBlock> batch;

  std::ifstream hex_file(hex_file_name);
  while (!hex_file.eof()) {
    std::string line;
    std::getline(hex_file, line);
    absl::StripAsciiWhitespace(&line);
    if (line.empty()) continue;

    auto machine_code = ParseHexString(line);
    // TODO(mtrofin): the absl dependency is about to be removed, this is
    // point-in-time
    if (!machine_code.has_value())
      return absl::InvalidArgumentError("cannot parse");

    llvm::Expected<std::vector<DisassembledInstruction>>
        disassembled_instructions =
            DisassembleAllInstructions((*llvm_support)->mc_disassembler(),
                                       (*llvm_support)->mc_instr_info(),
                                       (*llvm_support)->mc_register_info(),
                                       (*llvm_support)->mc_subtarget_info(),
                                       *inst_printer, 0, *machine_code);
    if (llvm::Error error = disassembled_instructions.takeError()) {
      return LlvmErrorToStatus(std::move(error));
    }
    std::vector<llvm::MCInst> mc_insts;
    mc_insts.reserve(disassembled_instructions->size());
    for (const DisassembledInstruction& disassembled_instruction :
         *disassembled_instructions) {
      mc_insts.push_back(std::move(disassembled_instruction.mc_inst));
    }

    if (batch.size() == max_num_blocks_per_batch) {
      if (absl::Status status = run_inference_for_batch(batch); status.ok()) {
        return status;
      }
      batch.clear();
    }
    batch.push_back(canonicalizer.BasicBlockFromMCInst(mc_insts));
  }
  // Process all remaining blocks.
  if (!batch.empty()) {
    if (absl::Status status = run_inference_for_batch(batch); status.ok()) {
      return status;
    }
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace gematria

int main(int argc, char* argv[]) {
  InitGoogle(argv[0], &argc, &argv, true);
  const absl::Status status =
      gematria::ProcessBasicBlocksFromCommandLineFlags();
  if (!status.ok()) {
    std::cerr << status.ToString();
    return 1;
  }
  return 0;
}
