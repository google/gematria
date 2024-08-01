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

#include <fstream>
#include <limits>

#include "gematria/datasets/process_and_filter_bbs_lib.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"

using namespace llvm;

cl::OptionCategory ProcessFilterCat("process_and_filter_bbs options",
                                    "The options specifically for controlling "
                                    "the behavior of process_and_filter_bbs.");

static cl::opt<std::string> InputFile(
    "input-file",
    cl::desc("Path to the input CSV file containing hex basic blocks"),
    cl::init(""), cl::cat(ProcessFilterCat));

static cl::opt<std::string> OutputFile(
    "output-file",
    cl::desc(
        "Path to the output CSV file with processed/filtered basic blocks"),
    cl::init(""), cl::cat(ProcessFilterCat));

static cl::opt<bool> FilterMemoryAccessingBlocks(
    "filter-memory-accessing-blocks",
    cl::desc("Whether or not to filter out blocks that access memory"),
    cl::init(false), cl::cat(ProcessFilterCat));

static cl::opt<unsigned> ReportProgressEvery(
    "report-progress-every",
    cl::desc("The interval at which to report progress in blocks"),
    cl::init(std::numeric_limits<unsigned>::max()), cl::cat(ProcessFilterCat));

int main(int Argc, char **Argv) {
  cl::ParseCommandLineOptions(Argc, Argv, "process_and_filter_bbs");

  ExitOnError ExitOnErr("process_and_filter_bbs error: ");

  gematria::BBProcessorFilter BBProcessor;

  unsigned LineCount = 0;

  std::ifstream InputFileStream(InputFile);
  std::ofstream OutputFileStream(OutputFile);
  for (std::string Line; std::getline(InputFileStream, Line);) {
    Expected<std::string> ProcessedBlockOrErr =
        BBProcessor.removeRiskyInstructions(Line, InputFile,
                                            FilterMemoryAccessingBlocks);
    if (!ProcessedBlockOrErr) ExitOnErr(ProcessedBlockOrErr.takeError());

    OutputFileStream << *ProcessedBlockOrErr << "\n";

    if (LineCount != 0 && LineCount % ReportProgressEvery == 0)
      dbgs() << "Finished block " << LineCount << "\n";

    ++LineCount;
  }
  return 0;
}
