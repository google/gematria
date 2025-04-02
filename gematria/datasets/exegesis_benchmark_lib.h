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
#include <optional>
#include <string_view>

#include "gematria/proto/execution_annotation.pb.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkCode.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkResult.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkRunner.h"
#include "llvm/tools/llvm-exegesis/lib/LlvmState.h"
#include "llvm/tools/llvm-exegesis/lib/SnippetRepetitor.h"

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

  llvm::Expected<llvm::exegesis::BenchmarkCode> processAnnotatedBlock(
      std::string_view BlockHex, const ExecutionAnnotations &Annotations);

  llvm::Expected<llvm::exegesis::Benchmark> benchmarkConfiguration(
      const llvm::exegesis::BenchmarkCode &BenchCode,
      const llvm::exegesis::SnippetRepetitor &Repetitor,
      unsigned int MinInstructions, std::optional<int> BenchmarkProcessCPU);

  llvm::Expected<double> benchmarkBasicBlock(
      const llvm::exegesis::BenchmarkCode &BenchCode,
      std::optional<int> BenchmarkProcessCPU);

 private:
  // This is a simple wrapper around functionality in ExegesisState that maps
  // optional values to a register or an error.
  llvm::Expected<llvm::MCRegister> getRegisterFromName(
      llvm::StringRef RegisterName);

  std::unique_ptr<llvm::MCContext> LLVMMCContext;
  std::unique_ptr<llvm::MCDisassembler> LLVMMCDisassembler;
  std::unique_ptr<llvm::MCInstPrinter> LLVMMCInstPrinter;

  std::unique_ptr<llvm::exegesis::BenchmarkRunner> BenchRunner;

  llvm::exegesis::LLVMState ExegesisState;
};

}  // namespace gematria
