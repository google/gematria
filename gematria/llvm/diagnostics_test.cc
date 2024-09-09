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

#include "gematria/llvm/diagnostics.h"

#include <memory>
#include <string_view>

#include "gematria/llvm/llvm_architecture_support.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"

namespace gematria {
namespace {

TEST(ScopedStringDiagnosticHandlerTest, TestDiagnostic) {
  static constexpr std::string_view kBufferContents = "#include <bar.h>\n\n";
  static constexpr std::string_view kBufferName = "foo.cc";
  static constexpr std::string_view kErrorMessage =
      "Something went terribly wrong";

  std::unique_ptr<LlvmArchitectureSupport> x86_64_ =
      LlvmArchitectureSupport::X86_64();
  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(kBufferContents, kBufferName),
      llvm::SMLoc());

  llvm::MCContext context(x86_64_->target_machine().getTargetTriple(),
                          &x86_64_->mc_asm_info(), &x86_64_->mc_register_info(),
                          &x86_64_->mc_subtarget_info(), &source_mgr);
  context.setMainFileName(kBufferName);

  ScopedStringDiagnosticHandler scoped_handler(context);
  context.reportError(llvm::SMLoc::getFromPointer(kBufferContents.data()),
                      kErrorMessage);

  EXPECT_THAT(scoped_handler.Get(),
              testing::AllOf(testing::HasSubstr(kBufferName),
                             testing::HasSubstr(kErrorMessage)));
}

}  // namespace
}  // namespace gematria
