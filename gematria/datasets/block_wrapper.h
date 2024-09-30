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

#ifndef RESEARCH_DEVTOOLS_EXEGESIS_GEMATRIA_BHIVE_BLOCK_WRAPPER_H_
#define RESEARCH_DEVTOOLS_EXEGESIS_GEMATRIA_BHIVE_BLOCK_WRAPPER_H_

#include <stdint.h>

#include "absl/types/span.h"

extern "C" {

// This is a function, defined in block_wrapper.S. But we're not going to call
// it directly, we're going to copy it after the block we're given, and execute
// the whole thing. So we declare it as a byte array instead.
extern const uint8_t gematria_prologue;

// Ditto.
extern const uint8_t gematria_epilogue;

// A separate symbol, also defined in block_wrapper.S, which gives the size of
// the function.
extern const uint64_t gematria_prologue_size;

// Ditto.
extern const uint64_t gematria_epilogue_size;
}

namespace gematria {

inline absl::Span<const uint8_t> GetPrologueCode() {
  return absl::MakeConstSpan(&gematria_prologue, gematria_prologue_size);
}

inline absl::Span<const uint8_t> GetEpilogueCode() {
  return absl::MakeConstSpan(&gematria_epilogue, gematria_epilogue_size);
}

struct RawX64Regs;

// This is the signature for the parameters the prologue code expects. It never
// returns, and makes no attempt to preserve registers or the stack frame.
typedef void (*WrappedFunc)(const RawX64Regs *initial_regs);

}  // namespace gematria

#endif  // RESEARCH_DEVTOOLS_EXEGESIS_GEMATRIA_BHIVE_BLOCK_WRAPPER_H_
