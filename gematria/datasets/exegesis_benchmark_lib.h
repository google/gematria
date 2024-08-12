// Copyright 2024 Google Inc.
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

#include <cstddef>
#include <memory>

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkCode.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkRunner.h"
#include "llvm/tools/llvm-exegesis/lib/LlvmState.h"

namespace gematria {

// ExegesisBenchmark is used for generating benchmark configurations for
// benchmarks and executing those benchmarks to get real-world data that
// can then be used for training Gematria models.
class ExegesisBenchmark {
 private:
  explicit ExegesisBenchmark(llvm::exegesis::LLVMState &&State);

 public:
  static llvm::Expected<std::unique_ptr<ExegesisBenchmark>> create();

  llvm::Expected<llvm::exegesis::BenchmarkCode> parseJSONBlock(
      const llvm::json::Object &BasicBlockJSON, size_t BlockIndex);

  llvm::Expected<double> benchmarkBasicBlock(
      const llvm::exegesis::BenchmarkCode &BenchCode);

 private:
  std::unique_ptr<llvm::MCContext> LLVMMCContext;
  std::unique_ptr<llvm::MCDisassembler> LLVMMCDisassembler;
  std::unique_ptr<llvm::MCInstPrinter> LLVMMCInstPrinter;

  std::unique_ptr<llvm::exegesis::BenchmarkRunner> BenchRunner;

  llvm::exegesis::LLVMState ExegesisState;
};

}  // namespace gematria
