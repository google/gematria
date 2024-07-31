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

#ifndef THIRD_PARTY_GEMATRIA_GEMATRIA_DATASETS_FIND_ACCESSED_ADDRS_H_
#define THIRD_PARTY_GEMATRIA_GEMATRIA_DATASETS_FIND_ACCESSED_ADDRS_H_

#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gematria/llvm/llvm_architecture_support.h"

namespace gematria {

// We need a register struct with each member directly encoded so that it has a
// predictible memory layout to pass into the prelude assembly which sets the
// registers before executing the block
struct RawX64Regs {
  int64_t rax;
  int64_t rbx;
  int64_t rcx;
  int64_t rdx;
  int64_t rsi;
  int64_t rdi;
  int64_t rsp;
  int64_t rbp;
  int64_t r8;
  int64_t r9;
  int64_t r10;
  int64_t r11;
  int64_t r12;
  int64_t r13;
  int64_t r14;
  int64_t r15;
};

struct RegisterAndValue {
  unsigned register_index;
  int64_t register_value;
};

struct BlockAnnotations {
  uintptr_t code_location;
  size_t block_size;
  uint64_t block_contents;
  std::vector<uintptr_t> accessed_blocks;
  std::vector<RegisterAndValue> initial_regs;
};

// Given a basic block of code, attempt to determine what addresses that code
// accesses. This is done by executing the code in a new process, so the code
// must match the architecture on which this function is executed.
absl::StatusOr<BlockAnnotations> FindAccessedAddrs(
    absl::Span<const uint8_t> basic_block,
    LlvmArchitectureSupport &llvm_arch_support);

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_DATASETS_FIND_ACCESSED_ADDRS_H_
