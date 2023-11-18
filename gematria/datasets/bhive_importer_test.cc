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

#include "gematria/datasets/bhive_importer.h"

#include <memory>

#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/testing/matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace gematria {
namespace {

class BHiveImporterTest : public ::testing::Test {
 protected:
  static constexpr absl::string_view kLlvmTriple = "x86_64-unknown-unknown";
  static constexpr absl::string_view kSourceName = "bhive: skl";

  static constexpr double kScaling = 1.0 / 100.0;

  void SetUp() override {
    x86_llvm_ = LlvmArchitectureSupport::X86_64();
    x86_canonicalizer_ =
        std::make_unique<X86Canonicalizer>(&x86_llvm_->target_machine());
    x86_bhive_importer_ =
        std::make_unique<BHiveImporter>(x86_canonicalizer_.get());
  }

  std::unique_ptr<LlvmArchitectureSupport> x86_llvm_;
  std::unique_ptr<Canonicalizer> x86_canonicalizer_;
  std::unique_ptr<BHiveImporter> x86_bhive_importer_;
};

using BHiveImporterDeathTest = BHiveImporterTest;

TEST_F(BHiveImporterTest, EmptyBlock) {
  EXPECT_THAT(
      x86_bhive_importer_->ParseBHiveCsvLine(kSourceName, ",0", 0, 1, kScaling),
      IsOkAndHolds(EqualsProto(R"pb(basic_block {}
                                    inverse_throughputs {
                                      source: "bhive: skl"
                                      inverse_throughput_cycles: 0
                                    })pb")));
}

TEST_F(BHiveImporterTest, OneInstruction) {
  EXPECT_THAT(x86_bhive_importer_->ParseBHiveCsvLine(
                  kSourceName, "4929d2,100.000000", 0, 1, 0.5),
              IsOkAndHolds(EqualsProto(
                  R"pb(basic_block {
                         machine_instructions {
                           assembly: "\tsubq\t%rdx, %r10"
                           machine_code: "I)\322"
                         }
                         canonicalized_instructions {
                           mnemonic: "SUB"
                           llvm_mnemonic: "SUB64rr"
                           output_operands { register_name: "R10" }
                           input_operands { register_name: "R10" }
                           input_operands { register_name: "RDX" }
                           implicit_output_operands { register_name: "EFLAGS" }
                         }
                       }
                       inverse_throughputs {
                         source: "bhive: skl"
                         inverse_throughput_cycles: 50
                       })pb")));
  EXPECT_THAT(x86_bhive_importer_->BasicBlockProtoFromMachineCodeHex("4929d2"),
              IsOkAndHolds(EqualsProto(
                  R"pb(machine_instructions {
                         assembly: "\tsubq\t%rdx, %r10"
                         machine_code: "I)\322"
                       }
                       canonicalized_instructions {
                         mnemonic: "SUB"
                         llvm_mnemonic: "SUB64rr"
                         output_operands { register_name: "R10" }
                         input_operands { register_name: "R10" }
                         input_operands { register_name: "RDX" }
                         implicit_output_operands { register_name: "EFLAGS" }
                       })pb")));
}

TEST_F(BHiveImporterTest, MultipleInstructions) {
  static constexpr absl::string_view kExpectedBasicBlockProto =
      R"pb(basic_block {
             machine_instructions {
               assembly: "\tsubq\t%rdx, %rbx"
               machine_code: "H)\323"
               address: 100
             }
             machine_instructions {
               assembly: "\tmovl\t108(%rsp), %eax"
               machine_code: "\213D$l"
               address: 103
             }
             machine_instructions {
               assembly: "\tmovl\t104(%rsp), %edx"
               machine_code: "\213T$h"
               address: 107
             }
             machine_instructions {
               assembly: "\tsarq\t$3, %rbx"
               machine_code: "H\301\373\003"
               address: 111
             }
             machine_instructions {
               assembly: "\tsubq\t%rdx, %rax"
               machine_code: "H)\320"
               address: 115
             }
             machine_instructions {
               assembly: "\tcmpq\t%rax, %rbx"
               machine_code: "H9\303"
               address: 118
             }
             canonicalized_instructions {
               mnemonic: "SUB"
               llvm_mnemonic: "SUB64rr"
               output_operands { register_name: "RBX" }
               input_operands { register_name: "RBX" }
               input_operands { register_name: "RDX" }
               implicit_output_operands { register_name: "EFLAGS" }
             }
             canonicalized_instructions {
               mnemonic: "MOV"
               llvm_mnemonic: "MOV32rm"
               output_operands { register_name: "EAX" }
               input_operands { memory { alias_group_id: 1 } }
               input_operands {
                 address { base_register: "RSP" displacement: 108 scaling: 1 }
               }
             }
             canonicalized_instructions {
               mnemonic: "MOV"
               llvm_mnemonic: "MOV32rm"
               output_operands { register_name: "EDX" }
               input_operands { memory { alias_group_id: 1 } }
               input_operands {
                 address { base_register: "RSP" displacement: 104 scaling: 1 }
               }
             }
             canonicalized_instructions {
               mnemonic: "SAR"
               llvm_mnemonic: "SAR64ri"
               output_operands { register_name: "RBX" }
               input_operands { register_name: "RBX" }
               input_operands { immediate_value: 3 }
               implicit_output_operands { register_name: "EFLAGS" }
             }
             canonicalized_instructions {
               mnemonic: "SUB"
               llvm_mnemonic: "SUB64rr"
               output_operands { register_name: "RAX" }
               input_operands { register_name: "RAX" }
               input_operands { register_name: "RDX" }
               implicit_output_operands { register_name: "EFLAGS" }
             }
             canonicalized_instructions {
               mnemonic: "CMP"
               llvm_mnemonic: "CMP64rr"
               input_operands { register_name: "RBX" }
               input_operands { register_name: "RAX" }
               implicit_output_operands { register_name: "EFLAGS" }
             }
           }
           inverse_throughputs {
             source: "bhive: skl"
             inverse_throughput_cycles: 2.07
           })pb";
  EXPECT_THAT(x86_bhive_importer_->ParseBHiveCsvLine(
                  kSourceName,
                  "4829d38b44246c8b54246848c1fb034829d04839c3,207.000000", 0, 1,
                  /*throughput_scaling=*/kScaling, /*base_address=*/100),
              IsOkAndHolds(EqualsProto(kExpectedBasicBlockProto)));
}

TEST_F(BHiveImporterTest, InvalidColumnIndices) {
  EXPECT_THAT(x86_bhive_importer_->ParseBHiveCsvLine(kSourceName, "4929d2", 0,
                                                     1, kScaling),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(BHiveImporterTest, InvalidColumnIndicesBig) {
  EXPECT_THAT(x86_bhive_importer_->ParseBHiveCsvLine(kSourceName,
                                                     "4929d2,a,b,c,5", 0, 10),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(BHiveImporterTest, InvalidColumnIndicesSame) {
  EXPECT_THAT(
      x86_bhive_importer_->ParseBHiveCsvLine(kSourceName, "4929d2,5", 0, 0),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(BHiveImporterTest, InvalidMachineCode) {
  // The binary code below is missing one byte at the end.
  EXPECT_THAT(x86_bhive_importer_->ParseBHiveCsvLine(kSourceName, "4929", 0, 1,
                                                     kScaling),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(x86_bhive_importer_->BasicBlockProtoFromMachineCodeHex("4929"),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(BHiveImporterTest, NonStandardColumns) {
  EXPECT_THAT(x86_bhive_importer_->ParseBHiveCsvLine(kSourceName, "a,b,,0", 2,
                                                     3, kScaling),
              IsOkAndHolds(EqualsProto(R"pb(basic_block {}
                                            inverse_throughputs {
                                              source: "bhive: skl"
                                              inverse_throughput_cycles: 0
                                            })pb")));
}

TEST_F(BHiveImporterTest, MIRDatasetBasicTest) {
  EXPECT_THAT(x86_bhive_importer_->LoadMIRModule("sample_dataset/data.mir"),
              IsOk());
  EXPECT_THAT(x86_bhive_importer_->ParseMIRCsvLine(kSourceName, "a,b,BB_13,2.37", 2,
                                                     3, kScaling),
              IsOk());
}

}  // namespace
}  // namespace gematria
