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

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gematria/datasets/extract_bbs_from_obj_lib.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("Input object file"),
                                          cl::init("-"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "extract_bbs_from_obj\n");

  ExitOnError ExitOnErr("extract_bbs_from_obj error: ");

  ErrorOr<std::unique_ptr<MemoryBuffer>> FileBufferOrErr =
      MemoryBuffer::getFile(InputFilename);

  if (!FileBufferOrErr) {
    ExitOnErr(make_error<StringError>("Failed to load the input file.",
                                      FileBufferOrErr.getError()));
  }

  std::vector<std::string> BasicBlocks =
      ExitOnErr(gematria::getBasicBlockHexValues(*FileBufferOrErr->get()));

  for (const std::string_view BasicBlock : BasicBlocks) {
    outs() << BasicBlock << "\n";
  }

  return 0;
}
