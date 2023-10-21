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

#ifndef THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_DIAGNOSTICS_H_
#define THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_DIAGNOSTICS_H_

#include <string>
#include <vector>

#include "llvm/MC/MCContext.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace gematria {

// A diagnostic handler that stores errors to a string and resets the context
// on destruction.
class ScopedStringDiagnosticHandler {
 public:
  explicit ScopedStringDiagnosticHandler(llvm::MCContext& context)
      : errors_os_(errors_), context_(context) {
    context_.setDiagnosticHandler([this](const llvm::SMDiagnostic& diag, bool,
                                         const llvm::SourceMgr&,
                                         std::vector<const llvm::MDNode*>&) {
      diag.print(nullptr, errors_os_);
    });
  }
  ~ScopedStringDiagnosticHandler() { context_.reset(); }

  // Returns the diagnostics collected by the handler.
  std::string Get() { return errors_os_.str(); }

 private:
  std::string errors_;
  llvm::raw_string_ostream errors_os_;
  llvm::MCContext& context_;
};

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_LLVM_DIAGNOSTICS_H_
