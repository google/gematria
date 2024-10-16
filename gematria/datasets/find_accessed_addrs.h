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

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/proto/execution_annotation.pb.h"

namespace gematria {

// Values specified explicitly, as the assembly prologue hardcodes these values.
// If they're changed, make sure to update block_wrapper.S
enum class VectorRegWidth : uint64_t {
  NONE = 0,
  XMM = 1,
  YMM = 2,
  ZMM = 3,
};

// We need a register struct with each member directly encoded so that it has a
// predictable memory layout to pass into the prelude assembly which sets the
// registers before executing the block. For convenience we make every element 8
// bytes large, so that there will be no padding and calculating offsets by hand
// is easy (as is required in our assembly prologue code).
struct RawX64Regs {
  VectorRegWidth max_vector_reg_width;
  uint64_t uses_upper_vector_regs;
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

  int64_t vector_regs[32];
};

// Given a basic block of code, attempt to determine what addresses that code
// accesses. This is done by executing the code in a new process, so the code
// must match the architecture on which this function is executed.
absl::StatusOr<ExecutionAnnotations> FindAccessedAddrs(
    absl::Span<const uint8_t> basic_block,
    LlvmArchitectureSupport &llvm_arch_support);

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_DATASETS_FIND_ACCESSED_ADDRS_H_
