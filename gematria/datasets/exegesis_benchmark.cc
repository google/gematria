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

#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "gematria/datasets/exegesis_benchmark_lib.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkCode.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkResult.h"
#include "llvm/tools/llvm-exegesis/lib/PerfHelper.h"
#include "llvm/tools/llvm-exegesis/lib/ResultAggregator.h"
#include "llvm/tools/llvm-exegesis/lib/TargetSelect.h"

using namespace llvm;
using namespace llvm::exegesis;

static cl::opt<std::string> AnnotatedBlocksJson(
    "annotated-blocks-json",
    cl::desc("Filename of the JSON file containing annotated basic blocks"),
    cl::init(""));

ExitOnError ExitOnErr("exegesis-benchmark error: ");

// TODO(boomanaiden154): The function below and the following template are taken
// from the llvm-exegesis code. These should be made generic and moved into
// Error.h in upstream LLVM most likely. Check Err. If it's in a failure state
// log the file error(s) and exit.
static void exitOnFileError(const Twine &FileName, Error Err) {
  if (Err) {
    ExitOnErr(createFileError(FileName, std::move(Err)));
  }
}

// Check E. If it's in a success state then return the contained value.
// If it's in a failure state log the file error(s) and exit.
template <typename T>
T exitOnFileError(const Twine &FileName, Expected<T> &&E) {
  exitOnFileError(FileName, E.takeError());
  return std::move(*E);
}

int main(int Argc, char *Argv[]) {
  cl::ParseCommandLineOptions(
      Argc, Argv, "Tool for benchmarking sets of annotated basic blocks");

  if (AnnotatedBlocksJson.empty())
    ExitOnErr(llvm::make_error<StringError>(
        errc::invalid_argument, "--annotated_blocks_json is required"));

  // LLVM Setup
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86Target();
  LLVMInitializeX86AsmPrinter();
  LLVMInitializeX86AsmParser();
  LLVMInitializeX86Disassembler();

  // Exegesis Setup
  InitializeX86ExegesisTarget();

  std::unique_ptr<gematria::ExegesisBenchmark> Benchmark =
      ExitOnErr(gematria::ExegesisBenchmark::create());

  if (pfm::pfmInitialize())
    ExitOnErr(llvm::make_error<StringError>(inconvertibleErrorCode(),
                                            "Failed to initialize libpfm"));

  auto JsonMemoryBuffer = ExitOnErr(
      errorOrToExpected(MemoryBuffer::getFile(AnnotatedBlocksJson, true)));

  json::Value ParsedAnnotatedBlocks = exitOnFileError(
      AnnotatedBlocksJson, json::parse(JsonMemoryBuffer->getBuffer()));

  for (size_t BlockIndex = 0;
       BlockIndex < ParsedAnnotatedBlocks.getAsArray()->size(); ++BlockIndex) {
    const llvm::json::Object *AnnotatedBlockObject =
        (*ParsedAnnotatedBlocks.getAsArray())[BlockIndex].getAsObject();
    if (!AnnotatedBlockObject)
      exitOnFileError(AnnotatedBlocksJson,
                      llvm::make_error<StringError>(
                          errc::invalid_argument,
                          "Malformed basic block: is not a JSON object"));

    BenchmarkCode BenchCode = exitOnFileError(
        AnnotatedBlocksJson,
        Benchmark->parseJSONBlock(*AnnotatedBlockObject, BlockIndex));

    double Throughput = exitOnFileError(
        AnnotatedBlocksJson, Benchmark->benchmarkBasicBlock(BenchCode));

    std::optional<StringRef> HexValue = AnnotatedBlockObject->getString("Hex");
    // The block has already been parsed previously, and thus should have thrown
    // an error if there is no hex value. Assert that this is the case here.
    assert(HexValue.has_value() &&
           "Expected block to already have been checked for a hex value.");

    outs() << *HexValue << "," << Throughput << "\n";
  }

  return 0;
}
