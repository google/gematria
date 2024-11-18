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

#include "third_party/gematria/gematria/datasets/find_accessed_addrs.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/memory/memory.h"
#include "third_party/absl/random/distributions.h"
#include "third_party/absl/random/random.h"
#include "third_party/absl/strings/str_format.h"
#include "third_party/absl/types/span.h"
#include "third_party/gematria/gematria/llvm/asm_parser.h"
#include "third_party/gematria/gematria/llvm/llvm_architecture_support.h"
#include "third_party/gematria/gematria/proto/execution_annotation.proto.h"
#include "third_party/gematria/gematria/testing/matchers.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/ADT/SmallString.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/ADT/SmallVector.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/IR/InlineAsm.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/MC/MCAsmInfo.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/MC/MCAssembler.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/MC/MCCodeEmitter.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/MC/MCContext.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/MC/MCFixup.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/MC/TargetRegistry.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Target/TargetMachine.h"

namespace gematria {
namespace {

using testing::ElementsAre;

uintptr_t AlignDown(uintptr_t x, size_t align) { return x - (x % align); }

class FindAccessedAddrsTest : public testing::Test {
 private:
  inline static std::unique_ptr<LlvmArchitectureSupport> llvm_arch_support_;

 protected:
  void SetUp() override {
    // Disable all find accessed addresses tests under sanitizers. The technique
    // used to detect memory accesses relies on low-level system hacks and it is
    // generally incompatible with sanitizers that expect nicely-behaving C++
    // code.
#if defined(__has_feature) && \
    (__has_feature(address_sanitizer) || __has_feature(memory_sanitizer))
    GTEST_SKIP() << "Not supported under sanitizers";
#endif
  }

  static void SetUpTestSuite() {
    llvm_arch_support_ = LlvmArchitectureSupport::X86_64();
  }

  std::string Assemble(std::string_view textual_assembly) {
    auto mc_insts_or = gematria::ParseAsmCodeFromString(
        llvm_arch_support_->target_machine(), textual_assembly,
        llvm::InlineAsm::AsmDialect::AD_Intel);
    CHECK_OK(mc_insts_or);
    const auto& mc_insts = *mc_insts_or;

    llvm::MCContext mc_context(
        llvm_arch_support_->target_machine().getTargetTriple(),
        &llvm_arch_support_->mc_asm_info(),
        &llvm_arch_support_->mc_register_info(),
        &llvm_arch_support_->mc_subtarget_info());
    const auto code_emitter =
        absl::WrapUnique(llvm_arch_support_->target().createMCCodeEmitter(
            llvm_arch_support_->mc_instr_info(), mc_context));

    llvm::SmallString<128> code;
    llvm::SmallVector<llvm::MCFixup> fixups;
    for (const auto& mc_inst : mc_insts) {
      code_emitter->encodeInstruction(mc_inst, code, fixups,
                                      llvm_arch_support_->mc_subtarget_info());
    }

    return std::string(code);
  }

  absl::StatusOr<ExecutionAnnotations> FindAccessedAddrsAsm(
      std::string_view textual_assembly) {
    auto code = Assemble(textual_assembly);
    auto span = absl::MakeConstSpan(
        reinterpret_cast<const uint8_t*>(code.data()), code.size());
    return FindAccessedAddrs(span, *llvm_arch_support_);
  }
};

TEST_F(FindAccessedAddrsTest, BasicMov) {
  EXPECT_THAT(FindAccessedAddrsAsm("mov [0x10000], eax"),
              IsOkAndHolds(Partially(EqualsProto("accessed_blocks: 0x10000"))));
}

TEST_F(FindAccessedAddrsTest, DISABLED_SingleAddressRandomTests) {
  absl::BitGen rng;

  // See https://www.kernel.org/doc/html/latest/x86/x86_64/mm.html for
  // a full description of the address space on Linux x86_64, which this test
  // is specific to.
  constexpr int64_t kMaxUserModeAddress = 0x0000'7fff'ffff'ffffL;

  // TODO(orodley): FindAccessedAddrs is currently quite slow, so we can't run
  // too many iterations without the test taking annoyingly long. Raise it later
  // when/if we've improved performance.
  for (int n = 0; n < 100; ++n) {
    // We only want to generate addresses within the user-mode virtual address
    // space.
    //
    // We use LogUniform so that we get a broad sampling of the address space.
    // If we used Uniform over such a large range we'd practically never get
    // addresses right towards the bottom of the range.
    auto addr = absl::LogUniform(rng, 0L, kMaxUserModeAddress);

    // mov can only take up to 32-bit immediate addresses, so we need to use a
    // temporary register.
    auto code = absl::StrFormat(R"asm(
      mov rax, %ld
      mov ebx, [rax]
    )asm",
                                addr);
    const absl::StatusOr<ExecutionAnnotations> result =
        FindAccessedAddrsAsm(code);
    ASSERT_OK(result.status());

    EXPECT_THAT(result->accessed_blocks(),
                ElementsAre(AlignDown(addr, result->block_size())));
    EXPECT_GT(result->code_start_address(), 0);
    EXPECT_LT(result->code_start_address(), kMaxUserModeAddress);
  }
}

TEST_F(FindAccessedAddrsTest, NoMemoryAccesses) {
  EXPECT_THAT(FindAccessedAddrsAsm("mov eax, ebx"),
              IsOkAndHolds(Partially(EqualsProto("accessed_blocks: []"))));
}

TEST_F(FindAccessedAddrsTest, MultipleAccesses) {
  EXPECT_THAT(FindAccessedAddrsAsm(R"asm(
    mov [0x10000], eax
    mov [0x20000], eax
  )asm"),
              IsOkAndHolds(Partially(
                  EqualsProto("accessed_blocks: [0x10000, 0x20000]"))));
}

TEST_F(FindAccessedAddrsTest, AccessFromRegister) {
  EXPECT_THAT(FindAccessedAddrsAsm(R"asm(
    mov [eax], eax
    mov [r11+r12], eax
  )asm"),
              IsOkAndHolds(Partially(
                  EqualsProto("accessed_blocks: [0x15000, 0x2a000]"))));
}

TEST_F(FindAccessedAddrsTest, DoubleIndirection) {
  EXPECT_THAT(FindAccessedAddrsAsm(R"asm(
    mov rax, [0x10000]
    mov rbx, [rax]
  )asm"),
              IsOkAndHolds(Partially(EqualsProto(
                  "accessed_blocks: [0x10000, 0x0000000800000000]"))));
}

TEST_F(FindAccessedAddrsTest, DivideByPointee) {
  EXPECT_THAT(FindAccessedAddrsAsm(R"asm(
    mov rbx, [rax]
    mov rdx, 0
    idiv rbx
    mov edx, 0
    mov ebx, [rcx]
    idiv ebx
  )asm"),
              IsOkAndHolds(Partially(EqualsProto("accessed_blocks: 0x15000"))));
}

TEST_F(FindAccessedAddrsTest, XmmRegister) {
  EXPECT_THAT(
      FindAccessedAddrsAsm(R"asm(
    movq rax, xmm0
    mov rax, [rax]
  )asm"),
      IsOkAndHolds(Partially(EqualsProto(R"pb(
        accessed_blocks: 0x15000
        initial_registers: { register_name: "XMM0" register_value: 0x15000 }
      )pb"))));
}

TEST_F(FindAccessedAddrsTest, YmmRegister) {
  EXPECT_THAT(
      FindAccessedAddrsAsm(R"asm(
    vpaddq ymm0, ymm0, ymm0
    movq rax, xmm0
    mov rax, [rax]
  )asm"),
      // This (and similar below) isn't the ideal result. Since we broadcast to
      // the whole register, it's redundant to include XMM0 as an input register
      // when we already have YMM0.
      // TODO(orodley): Fix this, either in getUsedRegisters, or by filtering
      // out such duplicates afterwards.
      IsOkAndHolds(Partially(EqualsProto(R"pb(
        accessed_blocks: 0x2a000
        initial_registers: { register_name: "XMM0" register_value: 0x15000 }
        initial_registers: { register_name: "YMM0" register_value: 0x15000 }
      )pb"))));
}

class FindAccessedAddrsAvx512Test : public FindAccessedAddrsTest {
 protected:
  void SetUp() override {
    bool host_supports_avx512 =
        __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");

    if (!host_supports_avx512) GTEST_SKIP() << "Host doesn't support AVX-512";

    FindAccessedAddrsTest::SetUp();
  }
};

TEST_F(FindAccessedAddrsAvx512Test, ZmmRegister) {
  EXPECT_THAT(
      FindAccessedAddrsAsm(R"asm(
    vpaddq zmm0, zmm0, zmm0
    movq rax, xmm0
    mov rax, [rax]
  )asm"),
      IsOkAndHolds(Partially(EqualsProto(R"pb(
        accessed_blocks: 0x2a000
        initial_registers: { register_name: "XMM0" register_value: 0x15000 }
        initial_registers: { register_name: "ZMM0" register_value: 0x15000 }
      )pb"))));
}

TEST_F(FindAccessedAddrsAvx512Test, UpperXmmRegister) {
  EXPECT_THAT(
      FindAccessedAddrsAsm(R"asm(
    vmovq rax, xmm21
    mov rax, [rax]
  )asm"),
      IsOkAndHolds(Partially(EqualsProto(R"pb(
        accessed_blocks: 0x15000
        initial_registers: { register_name: "XMM21" register_value: 0x15000 }
      )pb"))));
}

TEST_F(FindAccessedAddrsAvx512Test, UpperYmmRegister) {
  EXPECT_THAT(
      FindAccessedAddrsAsm(R"asm(
    vpaddq ymm30, ymm30, ymm30
    vmovq rax, xmm30
    mov rax, [rax]
  )asm"),
      IsOkAndHolds(Partially(EqualsProto(R"pb(
        accessed_blocks: 0x2a000
        initial_registers: { register_name: "XMM30" register_value: 0x15000 }
        initial_registers: { register_name: "YMM30" register_value: 0x15000 }
      )pb"))));
}

TEST_F(FindAccessedAddrsAvx512Test, UpperZmmRegister) {
  EXPECT_THAT(
      FindAccessedAddrsAsm(R"asm(
    vpaddq zmm18, zmm18, zmm18
    vmovq rax, xmm18
    mov rax, [rax]
  )asm"),
      IsOkAndHolds(Partially(EqualsProto(R"pb(
        accessed_blocks: 0x2a000
        initial_registers: { register_name: "XMM18" register_value: 0x15000 }
        initial_registers: { register_name: "ZMM18" register_value: 0x15000 }
      )pb"))));
}

class FindAccessedAddrsApxTest : public FindAccessedAddrsTest {
 protected:
  void SetUp() override {
    if (!__builtin_cpu_supports("apxf")) {
      GTEST_SKIP() << "Host doesn't support APX";
    }

    FindAccessedAddrsTest::SetUp();
  }
};

TEST_F(FindAccessedAddrsApxTest, UpperGpr) {
  EXPECT_THAT(
      FindAccessedAddrsAsm(R"asm(
    mov r23, [r30]
  )asm"),
      IsOkAndHolds(Partially(EqualsProto(R"pb(
        accessed_blocks: 0x15000
        initial_registers: { register_name: "R30" register_value: 0x15000 }
      )pb"))));
}

}  // namespace
}  // namespace gematria
