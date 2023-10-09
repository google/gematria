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

#include "gematria/basic_block/basic_block.h"
#include "gematria/granite/graph_builder_model_inference.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/utils/string.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/lite/model_builder.h"

namespace gematria {
namespace {

namespace cl = llvm::cl;

cl::opt<std::string> tflite_file(
    "gematria_tflite_file", cl::value_desc("tflite_file"),
    cl::desc("The path to the .tflite file that contains the trained model."));
cl::opt<std::string> basic_block_hex_file(
    "gematria_basic_block_hex_file", cl::value_desc("hex_file"),
    cl::desc(
        "The file from which the tool reads basic blocks in the hex format used"
        " in the BHive data set, one basic block per line."));
cl::opt<int> max_blocks_per_batch(
    "gematria_max_blocks_per_batch", cl::init(std::numeric_limits<int>::max()),
    cl::value_desc("num_blocks"),
    cl::desc("The maximal number of blocks per batch. When non-positive, all"
             " blocks are put into the same batch."));

void PrintPredictionsToStdout(
    const GraphBuilderModelInference::OutputType& predictions) {
  for (int i = 0; i < predictions.size(); ++i) {
    if (i > 0) std::cout << ",";
    std::cout << predictions[i];
  }
}

llvm::Error ProcessBasicBlocksFromCommandLineFlags() {
  constexpr char kLlvmTriple[] = "x86_64-unknown-unknown";
  llvm::Expected<std::unique_ptr<LlvmArchitectureSupport>> llvm_support =
      LlvmArchitectureSupport::FromTriple(kLlvmTriple, "", "");
  if (llvm::Error error = llvm_support.takeError()) return error;

  const std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(tflite_file.c_str());
  if (model == nullptr) {
    return llvm::createStringError(llvm::errc::io_error,
                                   "Could not load the TfLite model.");
  }
  llvm::Expected<std::unique_ptr<GraphBuilderModelInference>>
      expected_inference =
          GraphBuilderModelInference::FromTfLiteModel(model.get());
  if (llvm::Error error = expected_inference.takeError()) return error;
  GraphBuilderModelInference& inference = **expected_inference;

  X86Canonicalizer canonicalizer(&(*llvm_support)->target_machine());

  std::unique_ptr<llvm::MCInstPrinter> inst_printer =
      (*llvm_support)->CreateMCInstPrinter(0);

  const auto run_inference_for_batch =
      [&](const std::vector<BasicBlock>& batch) -> llvm::Error {
    inference.Reset();
    std::vector<bool> is_valid_block;
    is_valid_block.reserve(batch.size());
    for (const BasicBlock& block : batch) {
      is_valid_block.push_back(inference.AddBasicBlockToBatch(block));
      if (!is_valid_block.back()) {
        std::cerr << "Invalid basic block:\n" << block.ToString() << "\n";
      }
    }
    llvm::Expected<std::vector<GraphBuilderModelInference::OutputType>>
        predictions = inference.RunInference();
    if (llvm::Error error = predictions.takeError()) return error;

    int prediction_index = 0;
    for (const bool is_valid : is_valid_block) {
      if (is_valid) {
        PrintPredictionsToStdout((*predictions)[prediction_index++]);
      } else {
        std::cout << "Invalid block";
      }
      std::cout << std::endl;
    }
    return llvm::Error::success();
  };

  std::vector<BasicBlock> batch;

  std::ifstream hex_file(basic_block_hex_file);
  while (!hex_file.eof()) {
    std::string line;
    std::getline(hex_file, line);
    StripAsciiWhitespace(&line);
    if (line.empty()) continue;

    auto machine_code = ParseHexString(line);
    if (!machine_code.has_value()) {
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "Can't parse input line: %s",
                                     line.c_str());
    }

    llvm::Expected<std::vector<DisassembledInstruction>>
        disassembled_instructions =
            DisassembleAllInstructions((*llvm_support)->mc_disassembler(),
                                       (*llvm_support)->mc_instr_info(),
                                       (*llvm_support)->mc_register_info(),
                                       (*llvm_support)->mc_subtarget_info(),
                                       *inst_printer, 0, *machine_code);
    if (llvm::Error error = disassembled_instructions.takeError()) {
      return error;
    }
    std::vector<llvm::MCInst> mc_insts;
    mc_insts.reserve(disassembled_instructions->size());
    for (const DisassembledInstruction& disassembled_instruction :
         *disassembled_instructions) {
      mc_insts.push_back(std::move(disassembled_instruction.mc_inst));
    }

    if (batch.size() == max_blocks_per_batch) {
      if (llvm::Error error = run_inference_for_batch(batch)) {
        return error;
      }
      batch.clear();
    }
    batch.push_back(canonicalizer.BasicBlockFromMCInst(mc_insts));
  }
  // Process all remaining blocks.
  if (!batch.empty()) {
    if (llvm::Error error = run_inference_for_batch(batch)) {
      return error;
    }
  }

  return llvm::Error::success();
}

}  // namespace
}  // namespace gematria

int main(int argc, char* argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  llvm::Error error = gematria::ProcessBasicBlocksFromCommandLineFlags();
  if (error) {
    llvm::errs() << error;
    return 1;
  }
  return 0;
}
