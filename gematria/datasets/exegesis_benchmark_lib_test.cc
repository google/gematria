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

#include "gematria/datasets/exegesis_benchmark_lib.h"

#include <memory>
#include <string>

#include "gematria/proto/execution_annotation.pb.h"
#include "gematria/testing/llvm.h"
#include "gematria/testing/parse_proto.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/lib/Target/X86/MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/tools/llvm-exegesis/lib/BenchmarkCode.h"
#include "llvm/tools/llvm-exegesis/lib/PerfHelper.h"
#include "llvm/tools/llvm-exegesis/lib/TargetSelect.h"

namespace gematria {
namespace {

using namespace llvm;
using namespace llvm::exegesis;

using ::testing::_;
using ::testing::FieldsAre;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

class ExegesisBenchmarkTest : public testing::Test {
 protected:
  std::unique_ptr<ExegesisBenchmark> Benchmark;

 protected:
  ExegesisBenchmarkTest() : Benchmark(cantFail(ExegesisBenchmark::create())){};

  static void SetUpTestSuite() {
    // LLVM Setup
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86Target();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86AsmParser();
    LLVMInitializeX86Disassembler();

    // Exegesis Setup
    InitializeX86ExegesisTarget();

    pfm::pfmInitialize();
  }

  Expected<std::string> getErrorMessage(StringRef JSONBlock) {
    Expected<json::Value> BlockValue = json::parse(JSONBlock);
    if (!BlockValue) return BlockValue.takeError();

    Expected<BenchmarkCode> BenchCode =
        Benchmark->parseJSONBlock(*BlockValue->getAsObject(), 1);
    if (BenchCode)
      return llvm::make_error<StringError>(errc::interrupted,
                                           "Failed to get BenchmarkCode");

    std::string ErrorMessage;
    handleAllErrors(BenchCode.takeError(),
                    [&ErrorMessage](StringError &BenchCodeError) {
                      ErrorMessage = BenchCodeError.getMessage();
                    });

    return ErrorMessage;
  }

  Expected<double> benchmark(StringRef JSONBlock) {
    Expected<json::Value> BlockValue = json::parse(JSONBlock);
    if (!BlockValue) return BlockValue.takeError();

    Expected<BenchmarkCode> BenchCode =
        Benchmark->parseJSONBlock(*BlockValue->getAsObject(), 0);
    if (!BenchCode) return BenchCode.takeError();

    return Benchmark->benchmarkBasicBlock(*BenchCode);
  }
};

TEST_F(ExegesisBenchmarkTest, TestParseJSONBlock) {
  StringRef JSONBlock = R"json(
  {
    "Hex": "3b31",
    "LoopRegister": "RAX",
    "MemoryDefinitions": [
      {
        "Name": "MEM",
        "Size": 4096,
        "Value": 1
      }
    ],
    "MemoryMappings": [
      {
        "Address": 86016,
        "Value": "MEM"
      }
    ],
    "RegisterDefinitions": [
      {
        "Register": "RCX",
        "Value": 86016
      },
      {
        "Register": "RSI",
        "Value": 86016
      }
    ]
  }
  )json";

  Expected<json::Value> BlockValue = json::parse(JSONBlock);
  ASSERT_TRUE(static_cast<bool>(BlockValue));
  Expected<BenchmarkCode> BenchCode =
      Benchmark->parseJSONBlock(*BlockValue->getAsObject(), 1);

  EXPECT_THAT(BenchCode->Key.Instructions,
              UnorderedElementsAre(IsMCInst(X86::CMP32rm, _)));

  EXPECT_EQ(BenchCode->Key.LoopRegister, X86::RAX);

  EXPECT_THAT(BenchCode->Key.MemoryValues,
              UnorderedElementsAre(Pair("MEM", FieldsAre(1, 4096, 0))));

  EXPECT_THAT(BenchCode->Key.RegisterInitialValues,
              UnorderedElementsAre(FieldsAre(54, 86016), FieldsAre(60, 86016)));
}

TEST_F(ExegesisBenchmarkTest, TestParseJSONNoHex) {
  Expected<std::string> ErrorMessage = getErrorMessage("{}");
  ASSERT_TRUE(static_cast<bool>(ErrorMessage));

  EXPECT_EQ(*ErrorMessage,
            "Malformed basic block: Basic block at index 1 has no hex value");
}

TEST_F(ExegesisBenchmarkTest, TestParseJSONInvalidHex) {
  Expected<std::string> ErrorMessage = getErrorMessage(R"json(
  {
    "Hex": "INVALIDHEX",
    "LoopRegister": "RAX"
  }
  )json");
  ASSERT_TRUE(static_cast<bool>(ErrorMessage));

  EXPECT_EQ(*ErrorMessage,
            "Malformed basic block: Basic block at index 1 has an invalid hex "
            "value: INVALIDHEX");
}

TEST_F(ExegesisBenchmarkTest, TestParseJSONNoRegisterDefinitions) {
  Expected<std::string> ErrorMessage = getErrorMessage(R"json(
  {
    "Hex": "3b31",
    "LoopRegister": "RAX"
  }
  )json");
  ASSERT_TRUE(static_cast<bool>(ErrorMessage));

  EXPECT_EQ(*ErrorMessage,
            "Malformed basic block: Basic block at index 1 has no register "
            "definitions array");
}

TEST_F(ExegesisBenchmarkTest, TestParseJSONMissingRegisterIndexValue) {
  Expected<std::string> ErrorMessage = getErrorMessage(R"json(
  {
    "Hex": "3b31",
    "LoopRegister": "RAX",
    "RegisterDefinitions": [
      {
        "Value": 86016
      }
    ]
  }
  )json");
  ASSERT_TRUE(static_cast<bool>(ErrorMessage));

  EXPECT_EQ(*ErrorMessage,
            "Malformed register definition: Basic block at index 1 is missing "
            "a register number or value for register at index 0");
}

TEST_F(ExegesisBenchmarkTest, TestParseJSONMissingMemoryDefinitions) {
  Expected<std::string> ErrorMessage = getErrorMessage(R"json(
  {
    "Hex": "3b31",
    "LoopRegister": "RAX",
    "RegisterDefinitions": [
      {
        "Register": "RCX",
        "Value": 86016
      }
    ]
  }
  )json");
  ASSERT_TRUE(static_cast<bool>(ErrorMessage));

  EXPECT_EQ(*ErrorMessage,
            "Malformed basic block: Basic block at index 1 has no memory "
            "definitions array");
}

TEST_F(ExegesisBenchmarkTest, TestParseJSONMemoryDefinitionNotObject) {
  Expected<std::string> ErrorMessage = getErrorMessage(R"json(
  {
    "Hex": "3b31",
    "LoopRegister": "RAX",
    "RegisterDefinitions": [
      {
        "Register": "RCX",
        "Value": 86016
      }
    ],
    "MemoryDefinitions": [42]
  }
  )json");
  ASSERT_TRUE(static_cast<bool>(ErrorMessage));

  EXPECT_EQ(*ErrorMessage,
            "Malformed memory definition: Basic block at index 1 has a memory "
            "definition at index 0 that is not a JSON object");
}

TEST_F(ExegesisBenchmarkTest, TestParseJSONMemoryDefinitionMissingField) {
  Expected<std::string> ErrorMessage = getErrorMessage(R"json(
  {
    "Hex": "3b31",
    "LoopRegister": "RAX",
    "RegisterDefinitions": [
      {
        "Register": "RCX",
        "Value": 86016
      }
    ],
    "MemoryDefinitions": [
      {
        "Value": 1
      }
    ]
  }
  )json");
  ASSERT_TRUE(static_cast<bool>(ErrorMessage));

  EXPECT_EQ(*ErrorMessage,
            "Malformed memory definition: Basic block at index 1 has a memory "
            "definition at index 0 with no size, name, or value");
}

TEST_F(ExegesisBenchmarkTest, TestParseJSONMissingMemoryMappings) {
  Expected<std::string> ErrorMessage = getErrorMessage(R"json(
  {
    "Hex": "3b31",
    "LoopRegister": "RAX",
    "RegisterDefinitions": [
      {
        "Register": "RCX",
        "Value": 86016
      }
    ],
    "MemoryDefinitions": [
      {
        "Name": "MEM",
        "Size": 4096,
        "Value": 1
      }
    ]
  }
  )json");
  ASSERT_TRUE(static_cast<bool>(ErrorMessage));

  EXPECT_EQ(*ErrorMessage,
            "Malformed basic block: Basic block at index 1 has no memory "
            "mappings array");
}

TEST_F(ExegesisBenchmarkTest, TestParseJSONMemoryMappingNonObject) {
  Expected<std::string> ErrorMessage = getErrorMessage(R"json(
  {
    "Hex": "3b31",
    "LoopRegister": "RAX",
    "RegisterDefinitions": [
      {
        "Register": "RCX",
        "Value": 86016
      }
    ],
    "MemoryDefinitions": [
      {
        "Name": "MEM",
        "Size": 4096,
        "Value": 1
      }
    ],
    "MemoryMappings": [42]
  }
  )json");
  ASSERT_TRUE(static_cast<bool>(ErrorMessage));

  EXPECT_EQ(*ErrorMessage,
            "Malformed memory mapping: Basic block at index 1 has a memory "
            "mapping at index 0 which is not a JSON object");
}

TEST_F(ExegesisBenchmarkTest, TestParseJSONMemoryMappingMissingField) {
  Expected<std::string> ErrorMessage = getErrorMessage(R"json(
  {
    "Hex": "3b31",
    "LoopRegister": "RAX",
    "RegisterDefinitions": [
      {
        "Register": "RCX",
        "Value": 86016
      }
    ],
    "MemoryDefinitions": [
      {
        "Name": "MEM",
        "Size": 4096,
        "Value": 1
      }
    ],
    "MemoryMappings": [
      {
        "Value": "MEM"
      }
    ]
  }
  )json");
  ASSERT_TRUE(static_cast<bool>(ErrorMessage));

  EXPECT_EQ(*ErrorMessage,
            "Malformed memory mapping: Basic block at index 1 has a memory "
            "mapping at index 0 with no name or address");
}

TEST_F(ExegesisBenchmarkTest, TestBenchmarkAdd) {
  Expected<double> BenchmarkResult = benchmark(R"json(
  {
    "Hex": "4801c1",
    "LoopRegister": "RAX",
    "RegisterDefinitions": [
      {
        "Register": "HSI",
        "Value": 1
      }
    ],
    "MemoryDefinitions": [],
    "MemoryMappings": []
  }
  )json");

  EXPECT_LT(*BenchmarkResult, 10);
}

TEST_F(ExegesisBenchmarkTest, TestProcessAnnotatedBlock) {
  // Turn formatting off here so that clang-format does not clobber the
  // textual protobuf formatting below.
  // clang-format off
  const ExecutionAnnotations Annotations = ParseTextProto(R"pb(
    code_start_address: 0xff
    block_size: 4096
    block_contents: 0xff
    accessed_blocks: [0xaa, 0xbb]
    initial_registers [
      {
        register_name: "RAX"
        register_value: 5
      },
      {
        register_name: "RCX"
        register_value: 17
      }
    ]
  )pb");
  // clang-format on

  Expected<BenchmarkCode> BenchmarkConfiguration =
      Benchmark->processAnnotatedBlock("3b31", Annotations);
  ASSERT_TRUE(static_cast<bool>(BenchmarkConfiguration));

  EXPECT_THAT(BenchmarkConfiguration->Key.Instructions,
              UnorderedElementsAre(IsMCInst(X86::CMP32rm, _)));

  EXPECT_THAT(
      BenchmarkConfiguration->Key.RegisterInitialValues,
      UnorderedElementsAre(FieldsAre(X86::RAX, 5), FieldsAre(X86::RCX, 17)));

  EXPECT_THAT(BenchmarkConfiguration->Key.MemoryValues,
              UnorderedElementsAre(Pair("MEM", FieldsAre(0xff, 4096, 0))));

  EXPECT_THAT(
      BenchmarkConfiguration->Key.MemoryMappings,
      UnorderedElementsAre(FieldsAre(0xaa, "MEM"), FieldsAre(0xbb, "MEM")));

  EXPECT_EQ(BenchmarkConfiguration->Key.SnippetAddress, 0xff);
  EXPECT_EQ(BenchmarkConfiguration->Key.LoopRegister, MCRegister::NoRegister);
}

TEST_F(ExegesisBenchmarkTest, TestBenchmarkFromAnnotatedBlock) {
  // clang-format off
  const ExecutionAnnotations Annotations = ParseTextProto(R"pb(
    code_start_address: 0
    block_size: 4096
    block_contents: 34359738376
    accessed_blocks: 86016
    initial_registers: [
      {
        register_name: "RCX"
        register_value: 86016
      },
      {
        register_name: "RSI"
        register_value: 86016
      }
    ]
    loop_register: "RAX"
  )pb");
  // clang-format on

  Expected<BenchmarkCode> BenchmarkConfiguration =
      Benchmark->processAnnotatedBlock("3b31", Annotations);
  ASSERT_TRUE(static_cast<bool>(BenchmarkConfiguration));

  Expected<double> BenchmarkResult =
      Benchmark->benchmarkBasicBlock(*BenchmarkConfiguration);
  EXPECT_LT(*BenchmarkResult, 10);
}

}  // namespace
}  // namespace gematria
