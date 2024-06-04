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
namespace internal {

template <template <typename T> class Templ>
struct X64RegsTemplate {
  typename Templ<int64_t>::type rax;
  typename Templ<int64_t>::type rbx;
  typename Templ<int64_t>::type rcx;
  typename Templ<int64_t>::type rdx;
  typename Templ<int64_t>::type rsi;
  typename Templ<int64_t>::type rdi;
  typename Templ<int64_t>::type rsp;
  typename Templ<int64_t>::type rbp;
  typename Templ<int64_t>::type r8;
  typename Templ<int64_t>::type r9;
  typename Templ<int64_t>::type r10;
  typename Templ<int64_t>::type r11;
  typename Templ<int64_t>::type r12;
  typename Templ<int64_t>::type r13;
  typename Templ<int64_t>::type r14;
  typename Templ<int64_t>::type r15;
};

template <typename T>
struct Optional {
  using type = std::optional<T>;
};

template <class T>
struct type_identity {
  using type = T;
};

}  // namespace internal

// We need two versions of the registers struct:
// * one with each member optional, so that we can communicate to the caller the
//   subset of registers that are required to be set
// * one with each member directly encoded, so that it has a predictible memory
//   layout to pass into the prelude assembly which sets the registers before
//    executing the block
using RawX64Regs = internal::X64RegsTemplate<internal::type_identity>;
using X64Regs = internal::X64RegsTemplate<internal::Optional>;

struct AccessedAddrs {
  uintptr_t code_location;
  size_t block_size;
  uint64_t block_contents;
  std::vector<uintptr_t> accessed_blocks;
  X64Regs initial_regs;
};

// Given a basic block of code, attempt to determine what addresses that code
// accesses. This is done by executing the code in a new process, so the code
// must match the architecture on which this function is executed.
absl::StatusOr<AccessedAddrs> FindAccessedAddrs(
    absl::Span<const uint8_t> basic_block);

// Public for testing.
X64Regs FindReadRegs(const LlvmArchitectureSupport& llvm_arch_support,
                     absl::Span<const uint8_t> basic_block);

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_DATASETS_FIND_ACCESSED_ADDRS_H_
