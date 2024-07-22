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

#include "gematria/granite/graph_builder.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "gematria/basic_block/basic_block.h"
#include "gematria/basic_block/basic_block_protos.h"
#include "gematria/model/oov_token_behavior.h"
#include "gematria/proto/basic_block.pb.h"
#include "gematria/testing/parse_proto.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace gematria {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;

// Tokens used in the basic blocks in tests. For simplicity, we do not use the
// full set of x86-64 tokens.
constexpr absl::string_view kImmediateToken = "_IMMEDIATE_";
constexpr absl::string_view kFpImmediateToken = "_FP_IMMEDIATE_";
constexpr absl::string_view kAddressToken = "_ADDRESS_";
constexpr absl::string_view kMemoryToken = "_MEMORY_";
constexpr absl::string_view kUnknownToken = "_UNKNOWN_";
constexpr absl::string_view kTokens[] = {
    // 0
    kImmediateToken, kFpImmediateToken, kAddressToken, kMemoryToken, "LEA",
    // 5
    "MOV", "NOT", "R14", "R15", "RAX",
    // 10
    "RBX", "RCX", "RDI", kUnknownToken, "NOP", "LOCK"};

// Names of Instruction annotations used in tests.
const std::vector<std::string> kAnnotationNames{"cache_miss_freq",
                                                "other_annotation"};

int TokenIndex(absl::string_view token) {
  const auto it = std::find(std::begin(kTokens), std::end(kTokens), token);
  EXPECT_NE(it, std::end(kTokens)) << "Invalid token: " << token;
  return static_cast<int>(it - std::begin(kTokens));
}

class BasicBlockGraphBuilderTest : public testing::Test {
 protected:
  void CreateBuilder(OutOfVocabularyTokenBehavior out_of_vocabulary_behavior) {
    std::vector<std::string> tokens(std::begin(kTokens), std::end(kTokens));
    builder_ = std::make_unique<BasicBlockGraphBuilder>(
        std::move(tokens),
        /*immediate_token =*/kImmediateToken,
        /*fp_immediate_token =*/kFpImmediateToken,
        /*address_token =*/kAddressToken,
        /*memory_token =*/kMemoryToken,
        /*annotation_names=*/kAnnotationNames, out_of_vocabulary_behavior);
  }
  std::unique_ptr<BasicBlockGraphBuilder> builder_;
};

TEST_F(BasicBlockGraphBuilderTest, EmptyBlock) {
  CreateBuilder(OutOfVocabularyTokenBehavior::ReturnError());
  ASSERT_FALSE(builder_->AddBasicBlockFromInstructions({}));
  ASSERT_FALSE(builder_->AddBasicBlock(BasicBlock()));
}

TEST_F(BasicBlockGraphBuilderTest, SingleInstruction) {
  CreateBuilder(OutOfVocabularyTokenBehavior::ReturnError());
  ASSERT_TRUE(builder_->AddBasicBlock(BasicBlockFromProto(ParseTextProto(R"pb(
    canonicalized_instructions: {
      mnemonic: "LEA"
      llvm_mnemonic: "LEA64r"
      output_operands: { register_name: "RDI" }
      input_operands: {
        address: { base_register: "RBX" displacement: 8 scaling: 1 }
      }
    })pb"))));
  EXPECT_EQ(builder_->num_graphs(), 1);
  EXPECT_EQ(builder_->num_nodes(), 5);
  EXPECT_EQ(builder_->num_edges(), 4);
  EXPECT_EQ(builder_->num_node_tokens(), std::size(kTokens));
  EXPECT_THAT(builder_->num_nodes_per_block(), ElementsAre(5));
  EXPECT_THAT(builder_->num_edges_per_block(), ElementsAre(4));

  EXPECT_THAT(builder_->node_types(),
              ElementsAre(NodeType::kInstruction, NodeType::kAddressOperand,
                          NodeType::kRegister, NodeType::kImmediate,
                          NodeType::kRegister));
  EXPECT_THAT(builder_->node_features(),
              ElementsAre(TokenIndex("LEA"), TokenIndex(kAddressToken),
                          TokenIndex("RBX"), TokenIndex(kImmediateToken),
                          TokenIndex("RDI")));
  EXPECT_THAT(builder_->InstructionNodeMask(),
              ElementsAre(true, false, false, false, false));

  EXPECT_THAT(builder_->edge_senders(), ElementsAre(2, 3, 1, 0));
  EXPECT_THAT(builder_->edge_receivers(), ElementsAre(1, 1, 0, 4));
  EXPECT_THAT(builder_->edge_types(),
              ElementsAre(EdgeType::kAddressBaseRegister,
                          EdgeType::kAddressDisplacement,
                          EdgeType::kInputOperands, EdgeType::kOutputOperands));

  EXPECT_THAT(
      builder_->global_features(),
      ElementsAre(ElementsAre(1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0)));
}

TEST_F(BasicBlockGraphBuilderTest, SingleInstructionWithPrefix) {
  CreateBuilder(OutOfVocabularyTokenBehavior::ReturnError());
  ASSERT_TRUE(builder_->AddBasicBlock(BasicBlockFromProto(ParseTextProto(R"pb(
    canonicalized_instructions: {
      mnemonic: "NOT"
      llvm_mnemonic: "NOT64m"
      prefixes: "LOCK"
      output_operands: { memory: { alias_group_id: 1 } }
      input_operands: { memory: { alias_group_id: 1 } }
      input_operands: { address: { base_register: "R15" scaling: 1 } }
    })pb"))));
  EXPECT_EQ(builder_->num_graphs(), 1);
  EXPECT_EQ(builder_->num_nodes(), 6);
  EXPECT_EQ(builder_->num_edges(), 5);
  EXPECT_EQ(builder_->num_node_tokens(), std::size(kTokens));
  EXPECT_THAT(builder_->num_nodes_per_block(), ElementsAre(6));
  EXPECT_THAT(builder_->num_edges_per_block(), ElementsAre(5));

  EXPECT_THAT(builder_->node_types(),
              ElementsAre(NodeType::kInstruction, NodeType::kPrefix,
                          NodeType::kMemoryOperand, NodeType::kAddressOperand,
                          NodeType::kRegister, NodeType::kMemoryOperand));
  EXPECT_THAT(builder_->node_features(),
              ElementsAre(TokenIndex("NOT"), TokenIndex("LOCK"),
                          TokenIndex(kMemoryToken), TokenIndex(kAddressToken),
                          TokenIndex("R15"), TokenIndex(kMemoryToken)));
  EXPECT_THAT(builder_->InstructionNodeMask(),
              ElementsAre(true, false, false, false, false, false));

  EXPECT_THAT(builder_->edge_senders(), ElementsAre(1, 2, 4, 3, 0));
  EXPECT_THAT(builder_->edge_receivers(), ElementsAre(0, 0, 3, 0, 5));
  EXPECT_THAT(
      builder_->edge_types(),
      ElementsAre(EdgeType::kInstructionPrefix, EdgeType::kInputOperands,
                  EdgeType::kAddressBaseRegister, EdgeType::kInputOperands,
                  EdgeType::kOutputOperands));

  EXPECT_THAT(
      builder_->global_features(),
      ElementsAre(ElementsAre(0, 0, 1, 2, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1)));
}

TEST_F(BasicBlockGraphBuilderTest, SingleInstructionWithAnnotation) {
  CreateBuilder(OutOfVocabularyTokenBehavior::ReturnError());
  ASSERT_TRUE(builder_->AddBasicBlock(BasicBlockFromProto(ParseTextProto(R"pb(
    canonicalized_instructions: {
      mnemonic: "MOV"
      llvm_mnemonic: "MOV64rr"
      output_operands: { register_name: "RCX" }
      input_operands: { register_name: "RAX" }
      instruction_annotations: { name: "cache_miss_freq" value: 0.875 }
    })pb"))));
  EXPECT_EQ(builder_->num_graphs(), 1);
  EXPECT_EQ(builder_->num_nodes(), 3);
  EXPECT_EQ(builder_->num_edges(), 2);
  EXPECT_EQ(builder_->num_node_tokens(), std::size(kTokens));
  EXPECT_THAT(builder_->num_nodes_per_block(), ElementsAre(3));
  EXPECT_THAT(builder_->num_edges_per_block(), ElementsAre(2));

  EXPECT_THAT(builder_->node_types(),
              ElementsAre(NodeType::kInstruction, NodeType::kRegister,
                          NodeType::kRegister));
  EXPECT_THAT(
      builder_->node_features(),
      ElementsAre(TokenIndex("MOV"), TokenIndex("RAX"), TokenIndex("RCX")));
  EXPECT_THAT(builder_->InstructionNodeMask(), ElementsAre(true, false, false));

  EXPECT_THAT(builder_->edge_senders(), ElementsAre(1, 0));
  EXPECT_THAT(builder_->edge_receivers(), ElementsAre(0, 2));
  EXPECT_THAT(builder_->edge_types(),
              ElementsAre(EdgeType::kInputOperands, EdgeType::kOutputOperands));

  EXPECT_THAT(
      builder_->global_features(),
      ElementsAre(ElementsAre(0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0)));

  EXPECT_THAT(builder_->annotation_names(),
              ElementsAre("cache_miss_freq", "other_annotation"));
  EXPECT_THAT(builder_->instruction_annotations(),
              ElementsAre(ElementsAre(0.875, -1)));
}

TEST_F(BasicBlockGraphBuilderTest, InvalidMnemonic_ReturnError) {
  CreateBuilder(OutOfVocabularyTokenBehavior::ReturnError());
  EXPECT_FALSE(builder_->AddBasicBlock(BasicBlockFromProto(ParseTextProto(R"pb(
    canonicalized_instructions: {
      mnemonic: "ThisInstructionDoesNotExist"
      llvm_mnemonic: "LEA64r"
      output_operands: { register_name: "RDI" }
      input_operands: {
        address: { base_register: "RBX" displacement: 8 scaling: 1 }
      }
    })pb"))));

  EXPECT_EQ(builder_->num_graphs(), 0);
  EXPECT_EQ(builder_->num_nodes(), 0);
  EXPECT_EQ(builder_->num_edges(), 0);
  EXPECT_EQ(builder_->num_node_tokens(), std::size(kTokens));
  EXPECT_THAT(builder_->num_nodes_per_block(), ElementsAre());
  EXPECT_THAT(builder_->num_edges_per_block(), ElementsAre());
}

TEST_F(BasicBlockGraphBuilderTest, InvalidRegister_ReturnError) {
  CreateBuilder(OutOfVocabularyTokenBehavior::ReturnError());
  EXPECT_FALSE(builder_->AddBasicBlock(BasicBlockFromProto(ParseTextProto(R"pb(
    canonicalized_instructions: {
      mnemonic: "LEA"
      llvm_mnemonic: "LEA64r"
      output_operands: { register_name: "RUI" }
      input_operands: {
        address: { base_register: "RBX" displacement: 8 scaling: 1 }
      }
    })pb"))));
  EXPECT_EQ(builder_->num_graphs(), 0);
  EXPECT_EQ(builder_->num_nodes(), 0);
  EXPECT_EQ(builder_->num_edges(), 0);
  EXPECT_THAT(builder_->num_nodes_per_block(), IsEmpty());
  EXPECT_THAT(builder_->num_edges_per_block(), IsEmpty());
  EXPECT_THAT(builder_->node_types(), IsEmpty());
  EXPECT_THAT(builder_->edge_types(), IsEmpty());
  EXPECT_THAT(builder_->edge_senders(), IsEmpty());
  EXPECT_THAT(builder_->edge_receivers(), IsEmpty());
  EXPECT_THAT(builder_->global_features(), IsEmpty());
}

TEST_F(BasicBlockGraphBuilderTest, InvalidAddress_ReturnError) {
  CreateBuilder(OutOfVocabularyTokenBehavior::ReturnError());
  EXPECT_FALSE(builder_->AddBasicBlock(BasicBlockFromProto(ParseTextProto(R"pb(
    canonicalized_instructions: {
      mnemonic: "LEA"
      llvm_mnemonic: "LEA64r"
      output_operands: { register_name: "RCX" }
      input_operands: {
        address: { base_register: "RUX" displacement: 8 scaling: 1 }
      }
    })pb"))));
  EXPECT_EQ(builder_->num_graphs(), 0);
  EXPECT_EQ(builder_->num_nodes(), 0);
  EXPECT_EQ(builder_->num_edges(), 0);
  EXPECT_THAT(builder_->num_nodes_per_block(), IsEmpty());
  EXPECT_THAT(builder_->num_edges_per_block(), IsEmpty());
  EXPECT_THAT(builder_->node_types(), IsEmpty());
  EXPECT_THAT(builder_->edge_types(), IsEmpty());
  EXPECT_THAT(builder_->edge_senders(), IsEmpty());
  EXPECT_THAT(builder_->edge_receivers(), IsEmpty());
  EXPECT_THAT(builder_->global_features(), IsEmpty());
}

TEST_F(BasicBlockGraphBuilderTest, InvalidMnemonic_ReplaceToken) {
  CreateBuilder(OutOfVocabularyTokenBehavior::ReplaceWithToken(
      std::string(kUnknownToken)));
  ASSERT_TRUE(builder_->AddBasicBlock(BasicBlockFromProto(ParseTextProto(R"pb(
    canonicalized_instructions: {
      mnemonic: "ThisInstructionDoesNotExist"
      llvm_mnemonic: "LEA64r"
      output_operands: { register_name: "RDI" }
      input_operands: {
        address: { base_register: "RBX" displacement: 8 scaling: 1 }
      }
    })pb"))));
  EXPECT_EQ(builder_->num_graphs(), 1);
  EXPECT_EQ(builder_->num_nodes(), 5);
  EXPECT_EQ(builder_->num_edges(), 4);
  EXPECT_EQ(builder_->num_node_tokens(), std::size(kTokens));
  EXPECT_THAT(builder_->num_nodes_per_block(), ElementsAre(5));
  EXPECT_THAT(builder_->num_edges_per_block(), ElementsAre(4));

  EXPECT_THAT(builder_->node_types(),
              ElementsAre(NodeType::kInstruction, NodeType::kAddressOperand,
                          NodeType::kRegister, NodeType::kImmediate,
                          NodeType::kRegister));
  EXPECT_THAT(builder_->node_features(),
              ElementsAre(TokenIndex(kUnknownToken), TokenIndex(kAddressToken),
                          TokenIndex("RBX"), TokenIndex(kImmediateToken),
                          TokenIndex("RDI")));
  EXPECT_THAT(builder_->InstructionNodeMask(),
              ElementsAre(true, false, false, false, false));

  EXPECT_THAT(builder_->edge_senders(), ElementsAre(2, 3, 1, 0));
  EXPECT_THAT(builder_->edge_receivers(), ElementsAre(1, 1, 0, 4));
  EXPECT_THAT(builder_->edge_types(),
              ElementsAre(EdgeType::kAddressBaseRegister,
                          EdgeType::kAddressDisplacement,
                          EdgeType::kInputOperands, EdgeType::kOutputOperands));

  EXPECT_THAT(
      builder_->global_features(),
      ElementsAre(ElementsAre(1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0)));
}

TEST_F(BasicBlockGraphBuilderTest, InvalidRegister_ReplaceToken) {
  CreateBuilder(OutOfVocabularyTokenBehavior::ReplaceWithToken(
      std::string(kUnknownToken)));
  ASSERT_TRUE(builder_->AddBasicBlock(BasicBlockFromProto(ParseTextProto(R"pb(
    canonicalized_instructions: {
      mnemonic: "LEA"
      llvm_mnemonic: "LEA64r"
      output_operands: { register_name: "RegisterDoesNotExist" }
      input_operands: {
        address: { base_register: "RBX" displacement: 8 scaling: 1 }
      }
    })pb"))));
  EXPECT_EQ(builder_->num_graphs(), 1);
  EXPECT_EQ(builder_->num_nodes(), 5);
  EXPECT_EQ(builder_->num_edges(), 4);
  EXPECT_EQ(builder_->num_node_tokens(), std::size(kTokens));
  EXPECT_THAT(builder_->num_nodes_per_block(), ElementsAre(5));
  EXPECT_THAT(builder_->num_edges_per_block(), ElementsAre(4));

  EXPECT_THAT(builder_->node_types(),
              ElementsAre(NodeType::kInstruction, NodeType::kAddressOperand,
                          NodeType::kRegister, NodeType::kImmediate,
                          NodeType::kRegister));
  EXPECT_THAT(builder_->node_features(),
              ElementsAre(TokenIndex("LEA"), TokenIndex(kAddressToken),
                          TokenIndex("RBX"), TokenIndex(kImmediateToken),
                          TokenIndex(kUnknownToken)));
  EXPECT_THAT(builder_->InstructionNodeMask(),
              ElementsAre(true, false, false, false, false));

  EXPECT_THAT(builder_->edge_senders(), ElementsAre(2, 3, 1, 0));
  EXPECT_THAT(builder_->edge_receivers(), ElementsAre(1, 1, 0, 4));
  EXPECT_THAT(builder_->edge_types(),
              ElementsAre(EdgeType::kAddressBaseRegister,
                          EdgeType::kAddressDisplacement,
                          EdgeType::kInputOperands, EdgeType::kOutputOperands));

  EXPECT_THAT(
      builder_->global_features(),
      ElementsAre(ElementsAre(1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)));
}

TEST_F(BasicBlockGraphBuilderTest, InvalidAddress_ReplaceToken) {
  CreateBuilder(OutOfVocabularyTokenBehavior::ReplaceWithToken(
      std::string(kUnknownToken)));
  ASSERT_TRUE(builder_->AddBasicBlock(BasicBlockFromProto(ParseTextProto(R"pb(
    canonicalized_instructions: {
      mnemonic: "LEA"
      llvm_mnemonic: "LEA64r"
      output_operands: { register_name: "RDI" }
      input_operands: {
        address: {
          base_register: "RegisterDoesNotExist"
          displacement: 8
          scaling: 1
        }
      }
    })pb"))));
  EXPECT_EQ(builder_->num_graphs(), 1);
  EXPECT_EQ(builder_->num_nodes(), 5);
  EXPECT_EQ(builder_->num_edges(), 4);
  EXPECT_EQ(builder_->num_node_tokens(), std::size(kTokens));
  EXPECT_THAT(builder_->num_nodes_per_block(), ElementsAre(5));
  EXPECT_THAT(builder_->num_edges_per_block(), ElementsAre(4));

  EXPECT_THAT(builder_->node_types(),
              ElementsAre(NodeType::kInstruction, NodeType::kAddressOperand,
                          NodeType::kRegister, NodeType::kImmediate,
                          NodeType::kRegister));
  EXPECT_THAT(builder_->node_features(),
              ElementsAre(TokenIndex("LEA"), TokenIndex(kAddressToken),
                          TokenIndex(kUnknownToken),
                          TokenIndex(kImmediateToken), TokenIndex("RDI")));
  EXPECT_THAT(builder_->InstructionNodeMask(),
              ElementsAre(true, false, false, false, false));

  EXPECT_THAT(builder_->edge_senders(), ElementsAre(2, 3, 1, 0));
  EXPECT_THAT(builder_->edge_receivers(), ElementsAre(1, 1, 0, 4));
  EXPECT_THAT(builder_->edge_types(),
              ElementsAre(EdgeType::kAddressBaseRegister,
                          EdgeType::kAddressDisplacement,
                          EdgeType::kInputOperands, EdgeType::kOutputOperands));

  EXPECT_THAT(
      builder_->global_features(),
      ElementsAre(ElementsAre(1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0)));
}

// Tests that the instruction nodes within the basic block are connected through
// their operands when they refer to the same value. Also ensures annotated and
// non-annotated instructions are handled correctly when mixed.
TEST_F(BasicBlockGraphBuilderTest, MultipleInstructions) {
  CreateBuilder(OutOfVocabularyTokenBehavior::ReturnError());
  ASSERT_TRUE(builder_->AddBasicBlock(BasicBlockFromProto(ParseTextProto(R"pb(
    canonicalized_instructions: {
      mnemonic: "MOV"
      llvm_mnemonic: "MOV64rm"
      output_operands: { register_name: "R14" }
      input_operands: { memory: { alias_group_id: 1 } }
      input_operands: { address: { base_register: "R15" scaling: 1 } }
      instruction_annotations: { name: "cache_miss_freq" value: 0.9 }
    }
    canonicalized_instructions: {
      mnemonic: "MOV"
      llvm_mnemonic: "MOV64rm"
      output_operands: { register_name: "RAX" }
      input_operands: { memory: { alias_group_id: 1 } }
      input_operands: {
        address: { base_register: "R14" displacement: 112 scaling: 1 }
      }
      instruction_annotations: { name: "unused_annotation", value: 1 }
    }
    canonicalized_instructions: {
      mnemonic: "MOV"
      llvm_mnemonic: "MOV64rr"
      output_operands: { register_name: "RCX" }
      input_operands: { register_name: "RAX" }
      instruction_annotations: { name: "cache_miss_freq" value: 0.01 }
      instruction_annotations: { name: "other_annotation", value: 0.5 }
    }
    canonicalized_instructions: {
      mnemonic: "NOT"
      llvm_mnemonic: "NOT64r"
      output_operands: { register_name: "RCX" }
      input_operands: { register_name: "RCX" }
    })pb"))));

  // The expected number of nodes:
  //  - first instruction: MOV, R14, _MEMORY_, _ADDRESS_, R15.
  //  - second instruction: MOV, RAX, _ADDRESS_, _IMMEDIATE_ (displacement);
  //    R14 and _MEMORY_ are reused from the first instruction.
  //  - third instruction: MOV, RCX; RAX is reused from the second instruction.
  //  - fourth instruction: NOT, RCX (output); RCX (input) is reused from the
  //    third instruction.
  constexpr int kExpectedNumNodes = 5 + 4 + 2 + 2;
  constexpr int kExpectedNumEdges = 4 +  // First instruction.
                                    5 +  // Second instruction.
                                    2 +  // Third instruction.
                                    2 +  // Fourth instruction.
                                    3;   // Structural dependencies.
  EXPECT_EQ(builder_->num_graphs(), 1);
  EXPECT_EQ(builder_->num_node_tokens(), std::size(kTokens));
  EXPECT_EQ(builder_->num_nodes(), kExpectedNumNodes);
  EXPECT_THAT(builder_->num_nodes_per_block(), ElementsAre(kExpectedNumNodes));
  EXPECT_EQ(builder_->num_edges(), kExpectedNumEdges);
  EXPECT_THAT(builder_->num_edges_per_block(), ElementsAre(kExpectedNumEdges));
  EXPECT_THAT(builder_->node_types(),
              ElementsAre(NodeType::kInstruction, NodeType::kMemoryOperand,
                          NodeType::kAddressOperand, NodeType::kRegister,
                          NodeType::kRegister, NodeType::kInstruction,
                          NodeType::kAddressOperand, NodeType::kImmediate,
                          NodeType::kRegister, NodeType::kInstruction,
                          NodeType::kRegister, NodeType::kInstruction,
                          NodeType::kRegister));
  EXPECT_THAT(
      builder_->node_features(),
      ElementsAre(
          // First instruction.
          TokenIndex("MOV"), TokenIndex(kMemoryToken),
          TokenIndex(kAddressToken), TokenIndex("R15"), TokenIndex("R14"),
          // Second instruction.
          TokenIndex("MOV"), TokenIndex(kAddressToken),
          TokenIndex(kImmediateToken), TokenIndex("RAX"),
          // Third instruction.
          TokenIndex("MOV"), TokenIndex("RCX"),
          // Fourth instruction.
          TokenIndex("NOT"), TokenIndex("RCX")));
  EXPECT_THAT(builder_->InstructionNodeMask(),
              ElementsAre(  // First instruction.
                  true, false, false, false, false,
                  // Second instruction.
                  true, false, false, false,
                  // Third instruction.
                  true, false,
                  // Fourth instruction.
                  true, false));

  EXPECT_THAT(
      builder_->edge_types(),
      ElementsAre(EdgeType::kInputOperands, EdgeType::kAddressBaseRegister,
                  EdgeType::kInputOperands, EdgeType::kOutputOperands,
                  EdgeType::kStructuralDependency, EdgeType::kInputOperands,
                  EdgeType::kAddressBaseRegister,
                  EdgeType::kAddressDisplacement, EdgeType::kInputOperands,
                  EdgeType::kOutputOperands, EdgeType::kStructuralDependency,
                  EdgeType::kInputOperands, EdgeType::kOutputOperands,
                  EdgeType::kStructuralDependency, EdgeType::kInputOperands,
                  EdgeType::kOutputOperands));
  EXPECT_THAT(builder_->edge_senders(),
              ElementsAre(1, 3, 2, 0, 0, 1, 4, 7, 6, 5, 5, 8, 9, 9, 10, 11));
  EXPECT_THAT(builder_->edge_receivers(),
              ElementsAre(0, 2, 0, 4, 5, 5, 6, 6, 5, 8, 9, 9, 10, 11, 11, 12));

  EXPECT_THAT(builder_->instruction_annotations(),
              ElementsAre(ElementsAre(0.9, -1), ElementsAre(-1, -1),
                          ElementsAre(0.01, 0.5), ElementsAre(-1, -1)));
}

// Tests that nodes in basic blocks added through different AddBasicBlock()
// calls are not connected.
TEST_F(BasicBlockGraphBuilderTest, MultipleBasicBlocks) {
  CreateBuilder(OutOfVocabularyTokenBehavior::ReturnError());
  ASSERT_TRUE(builder_->AddBasicBlock(BasicBlockFromProto(ParseTextProto(R"pb(
    canonicalized_instructions: {
      mnemonic: "NOT"
      llvm_mnemonic: "NOT64r"
      output_operands: { register_name: "RCX" }
      input_operands: { register_name: "RCX" }
    })pb"))));
  ASSERT_TRUE(builder_->AddBasicBlock(BasicBlockFromProto(ParseTextProto(R"pb(
    canonicalized_instructions: {
      mnemonic: "NOT"
      llvm_mnemonic: "NOT64r"
      output_operands: { register_name: "RCX" }
      input_operands: { register_name: "RCX" }
    })pb"))));

  EXPECT_EQ(builder_->num_graphs(), 2);
  EXPECT_EQ(builder_->num_node_tokens(), std::size(kTokens));

  EXPECT_EQ(builder_->num_nodes(), 3 + 3);
  EXPECT_THAT(builder_->num_nodes_per_block(), ElementsAre(3, 3));

  EXPECT_EQ(builder_->num_edges(), 2 + 2);
  EXPECT_THAT(builder_->num_edges_per_block(), ElementsAre(2, 2));

  EXPECT_THAT(builder_->node_types(),
              ElementsAre(NodeType::kInstruction, NodeType::kRegister,
                          NodeType::kRegister, NodeType::kInstruction,
                          NodeType::kRegister, NodeType::kRegister));
  EXPECT_THAT(
      builder_->node_features(),
      ElementsAre(TokenIndex("NOT"), TokenIndex("RCX"), TokenIndex("RCX"),
                  TokenIndex("NOT"), TokenIndex("RCX"), TokenIndex("RCX")));
  EXPECT_THAT(builder_->edge_types(),
              ElementsAre(EdgeType::kInputOperands, EdgeType::kOutputOperands,
                          EdgeType::kInputOperands, EdgeType::kOutputOperands));

  EXPECT_THAT(builder_->edge_senders(), ElementsAre(1, 0, 4, 3));
  EXPECT_THAT(builder_->edge_receivers(), ElementsAre(0, 2, 3, 5));

  EXPECT_THAT(
      builder_->global_features(),
      ElementsAre(ElementsAre(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0),
                  ElementsAre(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0)));

  EXPECT_THAT(builder_->DeltaBlockIndex(), ElementsAre(0, 1));
}

TEST_F(BasicBlockGraphBuilderTest, TwoNops) {
  CreateBuilder(OutOfVocabularyTokenBehavior::ReturnError());
  ASSERT_TRUE(builder_->AddBasicBlock(BasicBlockFromProto(ParseTextProto(R"pb(
    canonicalized_instructions { mnemonic: "NOP" llvm_mnemonic: "NOOP" }
    canonicalized_instructions { mnemonic: "NOP" llvm_mnemonic: "NOOP" }
  )pb"))));
}

}  // namespace
}  // namespace gematria
