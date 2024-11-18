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

#include "third_party/absl/status/statusor.h"
#include "third_party/absl/types/span.h"
#include "third_party/gematria/gematria/llvm/llvm_architecture_support.h"
#include "third_party/gematria/gematria/proto/execution_annotation.proto.h"

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
  VectorRegWidth max_vector_reg_width;  // offset 0x0
  // If true, the code uses at least one of the 16 extra vector registers
  // defined in AVX-512. This is interpreted in combination with the max width.
  // For example, if max_vector_reg_width is XMM and uses_upper_vector_regs is
  // true, then the code uses XMM0-XMM31 but no YMM or ZMM registers.
  //
  // If this is false, then the latter 16 elements of vector_regs are unset and
  // should be ignored.
  uint64_t uses_upper_vector_regs;  // offset 0x8
  // If true, the code uses at least one of the 16 extra general purpose
  // registers defined in APX.
  //
  // If this is false, then the elements of apx_regs are unset and should be
  // ignored.
  uint64_t uses_apx_regs;  // offset 0x10
  int64_t rax;             // offset 0x18
  int64_t rbx;             // offset 0x20
  int64_t rcx;             // offset 0x28
  int64_t rdx;             // offset 0x30
  int64_t rsi;             // offset 0x38
  int64_t rdi;             // offset 0x40
  int64_t rsp;             // offset 0x48
  int64_t rbp;             // offset 0x50
  int64_t r8;              // offset 0x58
  int64_t r9;              // offset 0x60
  int64_t r10;             // offset 0x68
  int64_t r11;             // offset 0x70
  int64_t r12;             // offset 0x78
  int64_t r13;             // offset 0x80
  int64_t r14;             // offset 0x88
  int64_t r15;             // offset 0x90

  int64_t apx_regs[16];     // offset 0x98
  int64_t vector_regs[32];  // offset 0x118
};

// Given a basic block of code, attempt to determine what addresses that code
// accesses. This is done by executing the code in a new process, so the code
// must match the architecture on which this function is executed.
absl::StatusOr<ExecutionAnnotations> FindAccessedAddrs(
    absl::Span<const uint8_t> basic_block,
    LlvmArchitectureSupport &llvm_arch_support);

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_DATASETS_FIND_ACCESSED_ADDRS_H_
