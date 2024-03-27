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

// TODO(boomanaiden154): Refactor this to use LLVM file utilities.
#include <cmath>
#include <fstream>
#include <string>
#include <string_view>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string> InputFilePath1(
    "input-file-1", cl::desc("Path to the first input CSV."), cl::init(""));

static cl::opt<std::string> InputFilePath2(
    "input-file-2", cl::desc("Path to the second input CSV."), cl::init(""));

int main(int Argc, char **Argv) {
  cl::ParseCommandLineOptions(Argc, Argv, "compare_bhive_csv");

  ExitOnError ExitOnErr("compare_bhive_csv error: ");

  if (InputFilePath1.empty() || InputFilePath2.empty())
    ExitOnErr(llvm::make_error<StringError>(
        errc::invalid_argument, "Input file paths must not be empty"));

  std::ifstream InputFile1 = std::ifstream(InputFilePath1);
  std::ifstream InputFile2 = std::ifstream(InputFilePath2);

  double DeviationSum = 0.0;
  int FileCount = 0;

  for (std::string File1Line, File2Line; std::getline(InputFile1, File1Line) &&
                                         std::getline(InputFile2, File2Line);) {
    size_t Line1CommaIndex = File1Line.find(',');
    size_t Line2CommaIndex = File2Line.find(',');

    if (Line1CommaIndex == std::string::npos ||
        Line2CommaIndex == std::string::npos)
      ExitOnErr(llvm::make_error<StringError>(errc::invalid_argument,
                                              "No comma found on input line"));

    std::string_view ThroughputValue1String =
        std::string_view(File1Line).substr(Line1CommaIndex + 1,
                                           File1Line.size());

    std::string_view ThroughputValue2String =
        std::string_view(File2Line).substr(Line2CommaIndex + 1,
                                           File2Line.size());

    int ThroughputValue1 = 0;
    int ThroughputValue2 = 0;

    if (!to_integer(StringRef(ThroughputValue1String), ThroughputValue1) ||
        !to_integer(StringRef(ThroughputValue2String), ThroughputValue2)) {
      ExitOnErr(llvm::make_error<StringError>(
          errc::invalid_argument, "Failed to parse integer values"));
    }

    double Deviation =
        std::abs((double)ThroughputValue1 - (double)ThroughputValue2) /
        (double)ThroughputValue1;
    DeviationSum += Deviation;

    ++FileCount;
  }

  dbgs() << DeviationSum / (double)FileCount << "\n";

  return 0;
}
