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

#include "gematria/datasets/annotating_importer.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/proto/throughput.pb.h"
#include "gematria/testing/matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace gematria {
namespace {

using ::testing::ElementsAre;

// TODO(virajbshah): Consider adding a test that builds a binary from source,
// runs a `perf record` on it, and then runs `GetAnnotatedBasicBlockProtos`
// to ensure compatibilty with/prevent regressions against different versions
// of `perf` and `quipper`.
class AnnotatingImporterTest : public ::testing::Test {
 protected:
  static constexpr std::string_view kElfObjectFilepath =
      "com_google_gematria/gematria/testing/testdata/"
      "simple_x86_elf_object";
  static constexpr std::string_view kPerfDataFilepath =
      "com_google_gematria/gematria/testing/testdata/"
      "simple_x86_elf_object.perf.data";
  static constexpr absl::string_view kSourceName = "test: skl";

  std::unique_ptr<bazel::tools::cpp::runfiles::Runfiles> runfiles_;

  void SetUp() override {
    x86_llvm_ = LlvmArchitectureSupport::X86_64();
    x86_canonicalizer_ =
        std::make_unique<X86Canonicalizer>(&x86_llvm_->target_machine());
    x86_annotating_importer_ =
        std::make_unique<AnnotatingImporter>(x86_canonicalizer_.get());

    std::string error;
    runfiles_ = std::unique_ptr<bazel::tools::cpp::runfiles::Runfiles>(
        bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error));
    assert(runfiles_ != nullptr);
  }

  std::unique_ptr<LlvmArchitectureSupport> x86_llvm_;
  std::unique_ptr<Canonicalizer> x86_canonicalizer_;
  std::unique_ptr<AnnotatingImporter> x86_annotating_importer_;
};

TEST_F(AnnotatingImporterTest, AnnotatedBasicBlockProtosFromBinary) {
  absl::StatusOr<std::vector<BasicBlockWithThroughputProto>> protos =
      x86_annotating_importer_->GetAnnotatedBasicBlockProtos(
          runfiles_->Rlocation(std::string(kElfObjectFilepath)),
          runfiles_->Rlocation(std::string(kPerfDataFilepath)), kSourceName);

  EXPECT_TRUE(protos.ok());
  EXPECT_EQ(protos->size(), 1);
  EXPECT_THAT((*protos)[0], EqualsProto(R"pb(
                basic_block {
                  machine_instructions {
                    address: 18446744073709547787
                    assembly: "\tmovl\t%ecx, %edx"
                    machine_code: "\211\312"
                  }
                  machine_instructions {
                    address: 18446744073709547789
                    assembly: "\timull\t%edx, %edx"
                    machine_code: "\017\257\322"
                  }
                  machine_instructions {
                    address: 18446744073709547792
                    assembly: "\taddl\t%edx, %eax"
                    machine_code: "\001\320"
                  }
                  machine_instructions {
                    address: 18446744073709547794
                    assembly: "\tdecl\t%ecx"
                    machine_code: "\377\311"
                  }
                  canonicalized_instructions {
                    mnemonic: "MOV"
                    llvm_mnemonic: "MOV32rr"
                    output_operands { register_name: "EDX" }
                    input_operands { register_name: "ECX" }
                  }
                  canonicalized_instructions {
                    mnemonic: "IMUL"
                    llvm_mnemonic: "IMUL32rr"
                    output_operands { register_name: "EDX" }
                    input_operands { register_name: "EDX" }
                    input_operands { register_name: "EDX" }
                    implicit_output_operands { register_name: "EFLAGS" }
                  }
                  canonicalized_instructions {
                    mnemonic: "ADD"
                    llvm_mnemonic: "ADD32rr"
                    output_operands { register_name: "EAX" }
                    input_operands { register_name: "EAX" }
                    input_operands { register_name: "EDX" }
                    implicit_output_operands { register_name: "EFLAGS" }
                  }
                  canonicalized_instructions {
                    mnemonic: "DEC"
                    llvm_mnemonic: "DEC32r"
                    output_operands { register_name: "ECX" }
                    input_operands { register_name: "ECX" }
                    implicit_output_operands { register_name: "EFLAGS" }
                  }
                }
                inverse_throughputs {
                  source: "test: skl"
                  inverse_throughput_cycles: 1.532258064516129
                }
              )pb"));
}

}  // namespace
}  // namespace gematria
