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
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gematria/basic_block/basic_block.h"
#include "gematria/model/oov_token_behavior.h"

namespace gematria {
namespace {

constexpr BasicBlockGraphBuilder::NodeIndex kInvalidNode(-1);
constexpr BasicBlockGraphBuilder::TokenIndex kInvalidTokenIndex(-1);

std::unordered_map<std::string, BasicBlockGraphBuilder::TokenIndex> MakeIndex(
    std::vector<std::string> items) {
  std::unordered_map<std::string, BasicBlockGraphBuilder::TokenIndex> result;
  for (BasicBlockGraphBuilder::TokenIndex i = 0; i < items.size(); ++i) {
    const auto insertion_result = result.emplace(std::move(items[i]), i);
    if (!insertion_result.second) {
      // TODO(ondrasej): Make this return a status.
      std::cerr << "Duplicate item: '" << insertion_result.first->first << "'";
      std::abort();
    }
  }
  return result;
}

BasicBlockGraphBuilder::TokenIndex FindTokenOrDie(
    const std::unordered_map<std::string, BasicBlockGraphBuilder::TokenIndex>&
        tokens,
    const std::string& token) {
  return tokens.at(token);
}

template <typename MapType, typename KeyType, typename DefaultType>
typename MapType::mapped_type& LookupOrInsert(
    MapType& map, const KeyType& key, const DefaultType& default_value) {
  const auto [it, inserted] = map.insert({key, default_value});
  return it->second;
}

}  // namespace

#define EXEGESIS_ENUM_CASE(os, enum_value) \
  case enum_value:                         \
    os << #enum_value;                     \
    break

std::ostream& operator<<(std::ostream& os, NodeType node_type) {
  switch (node_type) {
    EXEGESIS_ENUM_CASE(os, NodeType::kInstruction);
    EXEGESIS_ENUM_CASE(os, NodeType::kRegister);
    EXEGESIS_ENUM_CASE(os, NodeType::kImmediate);
    EXEGESIS_ENUM_CASE(os, NodeType::kFpImmediate);
    EXEGESIS_ENUM_CASE(os, NodeType::kAddressOperand);
    EXEGESIS_ENUM_CASE(os, NodeType::kMemoryOperand);
    EXEGESIS_ENUM_CASE(os, NodeType::kPrefix);
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, EdgeType edge_type) {
  switch (edge_type) {
    EXEGESIS_ENUM_CASE(os, EdgeType::kStructuralDependency);
    EXEGESIS_ENUM_CASE(os, EdgeType::kInputOperands);
    EXEGESIS_ENUM_CASE(os, EdgeType::kOutputOperands);
    EXEGESIS_ENUM_CASE(os, EdgeType::kAddressBaseRegister);
    EXEGESIS_ENUM_CASE(os, EdgeType::kAddressIndexRegister);
    EXEGESIS_ENUM_CASE(os, EdgeType::kAddressSegmentRegister);
    EXEGESIS_ENUM_CASE(os, EdgeType::kAddressDisplacement);
    EXEGESIS_ENUM_CASE(os, EdgeType::kReverseStructuralDependency);
    EXEGESIS_ENUM_CASE(os, EdgeType::kInstructionPrefix);
  }
  return os;
}

#undef EXEGESIS_ENUM_CASE

BasicBlockGraphBuilder::AddBasicBlockTransaction::AddBasicBlockTransaction(
    BasicBlockGraphBuilder* graph_builder)
    : graph_builder_(*graph_builder),
      prev_num_nodes_per_block_size_(
          graph_builder->num_nodes_per_block_.size()),
      prev_num_edges_per_block_size_(
          graph_builder->num_edges_per_block_.size()),
      prev_node_types_size_(graph_builder->node_types_.size()),
      prev_node_features_size_(graph_builder->node_features_.size()),
      prev_edge_senders_size_(graph_builder->edge_senders_.size()),
      prev_edge_receivers_size_(graph_builder->edge_receivers_.size()),
      prev_edge_types_size_(graph_builder->edge_receivers_.size()),
      prev_global_features_size_(graph_builder->global_features_.size()) {}

BasicBlockGraphBuilder::AddBasicBlockTransaction::~AddBasicBlockTransaction() {
  if (!is_committed_) Rollback();
}

void BasicBlockGraphBuilder::AddBasicBlockTransaction::Commit() {
  is_committed_ = true;
}

#define GEMATRIA_CHECK_AND_RESIZE(vector_name)                              \
  do {                                                                      \
    const size_t original_size = prev_##vector_name##size_;                 \
    assert(original_size <= graph_builder_.vector_name.size() &&            \
           "The size of " #vector_name                                      \
           " has decreased. Did you call BasicBlockGraphBuilder::Reset()"); \
    graph_builder_.vector_name.resize(original_size);                       \
  } while (false)

void BasicBlockGraphBuilder::AddBasicBlockTransaction::Rollback() {
  assert(!is_committed_);
  GEMATRIA_CHECK_AND_RESIZE(num_nodes_per_block_);
  GEMATRIA_CHECK_AND_RESIZE(num_edges_per_block_);
  GEMATRIA_CHECK_AND_RESIZE(node_types_);
  GEMATRIA_CHECK_AND_RESIZE(node_features_);
  GEMATRIA_CHECK_AND_RESIZE(edge_senders_);
  GEMATRIA_CHECK_AND_RESIZE(edge_receivers_);
  GEMATRIA_CHECK_AND_RESIZE(edge_types_);
  GEMATRIA_CHECK_AND_RESIZE(global_features_);
}

#undef GEMATRIA_CHECK_AND_RESIZE

BasicBlockGraphBuilder::BasicBlockGraphBuilder(
    std::vector<std::string> node_tokens, std::string_view immediate_token,
    std::string_view fp_immediate_token, std::string_view address_token,
    std::string_view memory_token,
    OutOfVocabularyTokenBehavior
        out_of_vocabulary_behavior /* = ReturnError() */
    )
    : node_tokens_(MakeIndex(std::move(node_tokens))),
      // TODO(ondrasej): Remove the std::string conversions once we switch to
      // C++20 and std::unordered_map gains templated lookup functions.
      immediate_token_(
          FindTokenOrDie(node_tokens_, std::string(immediate_token))),
      fp_immediate_token_(
          FindTokenOrDie(node_tokens_, std::string(fp_immediate_token))),
      address_token_(FindTokenOrDie(node_tokens_, std::string(address_token))),
      memory_token_(FindTokenOrDie(node_tokens_, std::string(memory_token))),
      out_of_vocabulary_behavior_(out_of_vocabulary_behavior),
      replacement_token_(
          out_of_vocabulary_behavior.behavior_type() ==
                  OutOfVocabularyTokenBehavior::BehaviorType::kReturnError
              ? kInvalidTokenIndex
              : FindTokenOrDie(
                    node_tokens_,
                    out_of_vocabulary_behavior.replacement_token())) {}

bool BasicBlockGraphBuilder::AddBasicBlockFromInstructions(
    const std::vector<Instruction>& instructions) {
  if (instructions.empty()) return false;
  AddBasicBlockTransaction transaction(this);

  // Clear the maps that are maintained per basic block.
  register_nodes_.clear();
  alias_group_nodes_.clear();

  const int prev_num_nodes = num_nodes();
  const int prev_num_edges = num_edges();

  NodeIndex previous_instruction_node = kInvalidNode;
  for (const Instruction& instruction : instructions) {
    // Add the instruction node.
    const NodeIndex instruction_node =
        AddNode(NodeType::kInstruction, instruction.mnemonic);
    if (instruction_node == kInvalidNode) {
      return false;
    }

    // Store the annotation within the instruction annotations vector for later
    // use (inclusion in embeddings).
    // TODO(virajbshah): Annotations are better stored on `Instruction`s as
    // lists. Update this appropriately once that change is made.
    instruction_annotations_.push_back(
      std::vector<double>{instruction.cache_miss_frequency.value});

    // Add nodes for prefixes of the instruction.
    for (const std::string& prefix : instruction.prefixes) {
      const NodeIndex prefix_node = AddNode(NodeType::kPrefix, prefix);
      if (prefix_node == kInvalidNode) {
        return false;
      }
      AddEdge(EdgeType::kInstructionPrefix, prefix_node, instruction_node);
    }

    // Add a structural dependency edge from the previous instruction.
    if (previous_instruction_node >= 0) {
      AddEdge(EdgeType::kStructuralDependency, previous_instruction_node,
              instruction_node);
    }

    // Add edges for input operands. And nodes too, if necessary.
    for (const InstructionOperand& operand : instruction.input_operands) {
      if (!AddInputOperand(instruction_node, operand)) return false;
    }
    for (const InstructionOperand& operand :
         instruction.implicit_input_operands) {
      if (!AddInputOperand(instruction_node, operand)) return false;
    }

    // Add edges and nodes for output operands.
    for (const InstructionOperand& operand : instruction.output_operands) {
      if (!AddOutputOperand(instruction_node, operand)) return false;
    }
    for (const InstructionOperand& operand :
         instruction.implicit_output_operands) {
      if (!AddOutputOperand(instruction_node, operand)) return false;
    }

    previous_instruction_node = instruction_node;
  }

  global_features_.emplace_back(num_node_tokens(), 0);
  std::vector<int>& global_features = global_features_.back();
  for (NodeIndex i = prev_num_nodes; i < node_features_.size(); ++i) {
    ++global_features[node_features_[i]];
  }

  // Record the number of nodes and edges created for this graph.
  num_nodes_per_block_.push_back(num_nodes() - prev_num_nodes);
  num_edges_per_block_.push_back(num_edges() - prev_num_edges);

  transaction.Commit();
  return true;
}

void BasicBlockGraphBuilder::Reset() {
  num_nodes_per_block_.clear();
  num_edges_per_block_.clear();

  node_types_.clear();
  node_features_.clear();

  edge_senders_.clear();
  edge_receivers_.clear();
  edge_types_.clear();

  global_features_.clear();
}

bool BasicBlockGraphBuilder::AddInputOperand(
    NodeIndex instruction_node, const InstructionOperand& operand) {
  assert(instruction_node >= 0);
  assert(instruction_node < num_nodes());

  switch (operand.type()) {
    case OperandType::kRegister: {
      if (!AddDependencyOnRegister(instruction_node, operand.register_name(),
                                   EdgeType::kInputOperands)) {
        return false;
      }
    } break;
    case OperandType::kImmediateValue: {
      AddEdge(EdgeType::kInputOperands,
              AddNode(NodeType::kImmediate, immediate_token_),
              instruction_node);
    } break;
    case OperandType::kFpImmediateValue: {
      AddEdge(EdgeType::kInputOperands,
              AddNode(NodeType::kFpImmediate, fp_immediate_token_),
              instruction_node);
    } break;
    case OperandType::kAddress: {
      const NodeIndex address_node =
          AddNode(NodeType::kAddressOperand, address_token_);
      const AddressTuple& address_tuple = operand.address();
      if (!address_tuple.base_register.empty()) {
        if (!AddDependencyOnRegister(address_node, address_tuple.base_register,
                                     EdgeType::kAddressBaseRegister)) {
          return false;
        }
      }
      if (!address_tuple.index_register.empty()) {
        if (!AddDependencyOnRegister(address_node, address_tuple.index_register,
                                     EdgeType::kAddressIndexRegister)) {
          return false;
        }
      }
      if (!address_tuple.segment_register.empty()) {
        if (!AddDependencyOnRegister(address_node,
                                     address_tuple.segment_register,
                                     EdgeType::kAddressSegmentRegister)) {
          return false;
        }
      }
      if (address_tuple.displacement != 0) {
        AddEdge(EdgeType::kAddressDisplacement,
                AddNode(NodeType::kImmediate, immediate_token_), address_node);
      }
      // NOTE(ondrasej): For now, we explicitly ignore the scaling.
      AddEdge(EdgeType::kInputOperands, address_node, instruction_node);
    } break;
    case OperandType::kMemory: {
      NodeIndex& alias_group_node = LookupOrInsert(
          alias_group_nodes_, operand.alias_group_id(), kInvalidNode);
      if (alias_group_node == kInvalidNode) {
        alias_group_node = AddNode(NodeType::kMemoryOperand, memory_token_);
      }
      AddEdge(EdgeType::kInputOperands, alias_group_node, instruction_node);
    } break;
    case OperandType::kUnknown:
      // TODO(ondrasej): Return an error instead.
      std::cerr << "The operand proto is empty";
      std::abort();
  }
  return true;
}

bool BasicBlockGraphBuilder::AddOutputOperand(
    NodeIndex instruction_node, const InstructionOperand& operand) {
  assert(instruction_node >= 0);
  assert(instruction_node < num_nodes());

  switch (operand.type()) {
    case OperandType::kRegister: {
      const NodeIndex register_node =
          AddNode(NodeType::kRegister, operand.register_name());
      if (register_node == kInvalidNode) return false;
      AddEdge(EdgeType::kOutputOperands, instruction_node, register_node);
      register_nodes_[operand.register_name()] = register_node;
    } break;
    case OperandType::kImmediateValue:
    case OperandType::kFpImmediateValue:
    case OperandType::kAddress:
      std::cerr << "Immediate values, floating-point immediate values and "
                   "address expressions can't be output operands.";
      std::abort();
      break;
    case OperandType::kMemory: {
      const NodeIndex alias_group_node =
          AddNode(NodeType::kMemoryOperand, memory_token_);
      alias_group_nodes_[operand.alias_group_id()] = alias_group_node;
      AddEdge(EdgeType::kOutputOperands, instruction_node, alias_group_node);
    } break;
    case OperandType::kUnknown:
      // TODO(ondrasej): Return an error.
      std::cerr << "The operand proto is empty";
      std::abort();
  }
  return true;
}

bool BasicBlockGraphBuilder::AddDependencyOnRegister(
    NodeIndex dependent_node, const std::string& register_name,
    EdgeType edge_type) {
  NodeIndex& operand_node =
      LookupOrInsert(register_nodes_, register_name, kInvalidNode);
  if (operand_node == kInvalidNode) {
    // Add a node for the register if it doesn't exist. This also updates the
    // node index in `node_by_register`.
    operand_node = AddNode(NodeType::kRegister, register_name);
  }
  if (operand_node == kInvalidNode) return false;
  AddEdge(edge_type, operand_node, dependent_node);
  return true;
}

BasicBlockGraphBuilder::NodeIndex BasicBlockGraphBuilder::AddNode(
    NodeType node_type, TokenIndex token_index) {
  const NodeIndex new_node_index = num_nodes();
  node_types_.push_back(node_type);
  node_features_.push_back(token_index);
  return new_node_index;
}

BasicBlockGraphBuilder::NodeIndex BasicBlockGraphBuilder::AddNode(
    NodeType node_type, const std::string& token) {
  const auto it = node_tokens_.find(token);
  TokenIndex token_index = kInvalidTokenIndex;
  if (it != node_tokens_.end()) {
    token_index = it->second;
  } else {
    // TODO(ondrasej): Make this error message optional.
    std::cerr << "Unexpected node token: '" << token << "'";
    switch (out_of_vocabulary_behavior_.behavior_type()) {
      case OutOfVocabularyTokenBehavior::BehaviorType::kReturnError:
        return kInvalidNode;
      case OutOfVocabularyTokenBehavior::BehaviorType::kReplaceToken:
        token_index = replacement_token_;
    }
  }
  return AddNode(node_type, token_index);
}

void BasicBlockGraphBuilder::AddEdge(EdgeType edge_type, NodeIndex sender,
                                     NodeIndex receiver) {
  assert(sender >= 0);
  assert(sender < num_nodes());
  assert(receiver >= 0);
  assert(receiver < num_nodes());
  edge_senders_.push_back(sender);
  edge_receivers_.push_back(receiver);
  edge_types_.push_back(edge_type);
}

std::vector<int> BasicBlockGraphBuilder::EdgeFeatures() const {
  std::vector<int> edge_features(num_edges());
  for (int i = 0; i < num_edges(); ++i) {
    edge_features[i] = static_cast<int>(edge_types_[i]);
  }
  return edge_features;
}

std::vector<bool> BasicBlockGraphBuilder::InstructionNodeMask() const {
  std::vector<bool> instruction_node_mask(num_nodes());
  for (NodeIndex i = 0; i < num_nodes(); ++i) {
    instruction_node_mask[i] = node_types_[i] == NodeType::kInstruction;
  }
  return instruction_node_mask;
}

std::vector<int> BasicBlockGraphBuilder::DeltaBlockIndex() const {
  const int num_instructions = static_cast<int>(std::count(
      node_types_.begin(), node_types_.end(), NodeType::kInstruction));
  std::vector<int> delta_block_index;
  delta_block_index.reserve(num_instructions);
  int block = -1;
  int block_end = 0;
  for (int node = 0; node < node_types_.size(); ++node) {
    if (node_types_[node] != NodeType::kInstruction) continue;
    while (node >= block_end && block < num_graphs()) {
      block++;
      block_end += num_nodes_per_block_[block];
    }
    delta_block_index.push_back(block);
  }
  assert(block == num_graphs() - 1);
  assert(block_end == num_nodes());
  assert(delta_block_index.size() == num_instructions);
  return delta_block_index;
}

namespace {
template <typename Container>
void StrAppendList(std::stringstream& buffer, std::string_view list_name,
                   const Container& items) {
  buffer << list_name << " = [";
  bool first = true;
  for (const auto& item : items) {
    if (!first) {
      buffer << ",";
      first = false;
    }
    buffer << item;
  }
  buffer << "]\n";
}
}  // namespace

std::string BasicBlockGraphBuilder::DebugString() const {
  std::stringstream buffer;

  buffer << "num_graphs = " << num_graphs() << "\n";
  buffer << "num_nodes = " << num_nodes() << "\n";
  buffer << "num_edges = " << num_edges() << "\n";
  buffer << "num_node_tokens = " << num_node_tokens() << "\n";
  StrAppendList(buffer, "num_nodes_per_block", num_nodes_per_block());
  StrAppendList(buffer, "num_edges_per_block", num_edges_per_block());
  StrAppendList(buffer, "node_types", node_types());
  StrAppendList(buffer, "edge_senders", edge_senders());
  StrAppendList(buffer, "edge_receivers", edge_receivers());
  StrAppendList(buffer, "edge_types", edge_types());
  StrAppendList(buffer, "InstructionNodeMask", InstructionNodeMask());
  StrAppendList(buffer, "DeltaBlockIndex", DeltaBlockIndex());
  return buffer.str();
}

}  // namespace gematria
