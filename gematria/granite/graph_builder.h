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

// Contains code that efficiently converts basic blocks from the format used by
// Gematria to a graph format suitable for processing with TensorFlow and the
// graph_nets library [1]. See the paper [2] for more details on the graph data
// structure used by the graph_nets library. The code in this module is intended
// to be used primarily from Python via a pybind11 wrapper.
//
// Each basic block is transformed into a single graph that covers the data flow
// in the basic block as well as the layout structure of the basic block. The
// node and global feature vectors created by the graph builder are based on the
// tokens from the canonicalized representation of the instructions in the basic
// blocks processed by the graph builder. The vocabulary of all tokens must be
// provided during the construction of the graph builder. The feature "vector"
// of each node is a single integer scalar, which is the index of the
// token associated with the node in the token vocabulary. See below for more
// information on the relation between the nodes of the graph and the tokens.
//
// The basic blocks are transformed into a graph as follows:
//  - The basic block is represented as a collection of nodes that represent the
//    instructions in the basic block and their operands. The edges represent
//    the relationships between the instructions and the operands.
//  - There are three basic types of nodes with several sub-types: instruction
//    nodes that represent instructions in the basic block, value nodes that
//    represent the data consumed and produced in the basic block, and address
//    computation nodes that represent the address computation within another
//    instruction.
//  - Each instruction is represented as a single (instruction) node of type
//    NodeType::kInstruction. The feature of the node is the index of the
//    mnemonic of the instruction in the vocabulary of tokens. In the following
//    text, we unify the instruction and the node that represents the
//    instruction.
//    All instruction nodes in the graph are connected using edges of type
//    EdgeType::kStructuralDependency into a path that represents the order of
//    of the instructions in the basic block.
//  - The value nodes represent the operands of the instruction. They are value
//    oriented, i.e. when two different values are written to the same storage
//    location, each is represented by a different node in the graph. A value
//    nodes has zero or one incoming edge of type EdgeType::kOutputOperands from
//    the instruction that produces the value, and zero or more outgoing edges
//    of type EdgeType::kInputOperands to instructions and address computation
//    nodes that consume the value.
//
//    There are the following types of value nodes:
//    - Register nodes (NodeType::kRegister) represent values stored in
//      registers of the CPU. The feature of the node is the index of the name
//      of the register in the token vocabulary.
//    - Immediate value and floating-point immediate value nodes
//      (NodeType::kImmediate and NodeType::kFpImmediate) represent immediate
//      values in the code. Each immediate value in the code is represented by
//      its own node. These nodes never have incoming edges. All immediate value
//      nodes use the same value provided during the initialization of the graph
//      builder as their feature. Same for the floating-point immediate value
//      nodes.
//    - Memory nodes (NodeType::kMemoryOperand) represent values stored in the
//      memory of the computer. All memory operands use the same feature value,
//      provided during the initialization of the basic block graph builder.
//  - The address computation represent the x86-64 address computation using the
//    ModR/M and SIB bytes. They are represented by a single node of type
//    NodeType::kAddressOperand that is connected to the instruction using the
//    address (using an EdgeType::kInputOperands edge). The components of the
//    address computation are nodes representing the registers and the immediate
//    values used in the address computation, connected to the address
//    computation node using edges of types EdgeType::kAddressBaseRegister,
//    EdgeType::kAddressIndexRegister, and EdgeType::kAddressDisplacement; each
//    of these edge types can be used at most once; the base and index register
//    edges can be used only with register nodes, and the displacement edge can
//    be used only with an immediate value node.
//
// [1] https://github.com/deepmind/graph_nets
// [2] Battaglia, Peter W., et al., Relational inductive biases, deep learning,
//     and graph networks, https://arxiv.org/abs/1806.01261
//
// TODO(ondrasej): Move the description to a g3doc file with examples of the
// conversion process.

#ifndef GEMATRIA_GRANITE_GRAPH_BUILDER_H_
#define GEMATRIA_GRANITE_GRAPH_BUILDER_H_

#include <cstddef>
#include <ostream>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gematria/basic_block/basic_block.h"
#include "gematria/model/oov_token_behavior.h"

namespace gematria {

// The types of nodes created by the BasicBlockGraphBuilder class. See the
// documentation of the BasicBlockGraphBuilder class for more information on the
// types of nodes used by the class.
enum class NodeType {
  kInstruction = 0,
  kRegister = 1,
  kImmediate = 2,
  kFpImmediate = 3,
  kAddressOperand = 4,
  kMemoryOperand = 5,
  kPrefix = 6,
};

// The types of edges created by the BasicBlockGraphBuilder class. See the
// documentation of the BasicBlockGraphBuilder class for more information on the
// types of edges used by the class.
enum class EdgeType {
  kStructuralDependency = 0,
  kInputOperands = 1,
  kOutputOperands = 2,
  kAddressBaseRegister = 3,
  kAddressIndexRegister = 4,
  kAddressSegmentRegister = 5,
  kAddressDisplacement = 6,
  // TODO(ondrasej): Remove this value after the experiments for the Granite
  // paper are completed. This value is not used, but it affects the size of an
  // embedding vector table; removing it would change the size of this table and
  // it would invalidate existing checkpoints.
  kReverseStructuralDependency = 7,
  kInstructionPrefix = 8,
};

std::ostream& operator<<(std::ostream& os, NodeType node_type);
std::ostream& operator<<(std::ostream& os, EdgeType edge_type);

// The basic block graph builder class. See the top-level comment for more
// information on the format of the graphs produced by this file.
class BasicBlockGraphBuilder {
 public:
  // The types of indices of nodes in the graph.
  using NodeIndex = int;
  // The index of a string token of a node. This is used as the feature
  // "vector" of the node, enabling the use of embedding lookup tables.
  using TokenIndex = int;
  // TODO(ondrasej): Consider using stronly typed integers.

  // Creates a new instance of the basic block builder for the given vocabulary.
  //  - node_tokens: the list of possible node tokens (tokens), i.e.
  //    instruction mnemonics, register names, and other tokens associated with
  //    nodes in the basic block graphs.
  //  - immediate_token: the token associated with nodes that represent an
  //    immediate value.
  //  - fp_immediate_token: the token associated with nodes that represent
  //    floating-point immediate values.
  //  - address_token: the token associated with nodes that represent
  //    address computation.
  //  - memory_token: the token associated with nodes that represent memory
  //    accesses.
  //  - annotation_names: the set of names of annotations to be used.
  //    Annotations with names belonging to this list will be stored and
  //    available for use, the rest will be discarded. All instructions need
  //    not have all annotations corresponding to the elements of this list,
  //    i.e. missing annotations will be handled.
  //  - unknown_token_behavior and unknown_token: controls for the behavior of
  //    the basic block graph builder when it encounters an unknown token when
  //    adding new basic blocks to the builder. When unknown_node_behavior is
  //    kReplaceToken, the unknown token is replaced with `unknown_token`;
  //    otherwise, unknown_token is ignored.
  //  - unknown_token_behavior: determines how the graph builder and
  // The values of immediate_token, fp_immediate_token, address_token and
  // memory_token must appear in node_tokens.
  BasicBlockGraphBuilder(
      std::vector<std::string> node_tokens, std::string_view immediate_token,
      std::string_view fp_immediate_token, std::string_view address_token,
      std::string_view memory_token,
      std::set<std::string> annotation_names = std::set<std::string>(),
      OutOfVocabularyTokenBehavior out_of_vocabulary_behavior =
          OutOfVocabularyTokenBehavior::ReturnError());

  // Adds a basic block to the graph builder.
  //
  // Returns true when the block was successfully added; returns false when the
  // method encountered an unknown token and the unknown token behavior is not
  // kReplaceToken or when the basic block does not contain any instructions.
  // When this happens, the graph builder is left in the previous state, i.e. no
  // basic block is added to it.
  bool AddBasicBlock(const BasicBlock& block) {
    return AddBasicBlockFromInstructions(block.instructions);
  }
  // A version of AddBasicBlock that takes the list of instructions in the basic
  // block instead of the basic block object itself.
  bool AddBasicBlockFromInstructions(
      const std::vector<Instruction>& instructions);

  // Resets the graph builder so that it can be used to create a new graph from
  // scratch.
  void Reset();

  // Returns the number of graphs in the batch. This corresponds to the number
  // of successful calls to AddBasicBlock() since the last call to Reset().
  int num_graphs() const {
    return static_cast<int>(num_nodes_per_block_.size());
  }

  // Returns the number of nodes in the current batch.
  int num_nodes() const { return static_cast<int>(node_types_.size()); }

  // Returns the number of edges in the current batch.
  int num_edges() const { return static_cast<int>(edge_senders_.size()); }

  // Returns the number of different tokens corresponding to nodes of the graph.
  int num_node_tokens() const { return static_cast<int>(node_tokens_.size()); }

  // The following getters provide access to the graphs in the current batch.
  // The data structures and the format of the data match the format used by the
  // graph_nets.GraphsTuple class that is fed to TensorFlow when processing the
  // batch.

  // TODO(ondrasej): For simplicity, this class produces the data as lists
  // (which get converted by pybind11 into Python lists), and the conversion to
  // NumPy arrays required by TensorFlow must be done from Python. If this
  // proves to be slowing down the computation, we could create optimize this by
  // creating NumPy arrays directly from the C++ code.

  // The number of nodes for each basic block in the batch. Corresponds to
  // `GraphsTuple.n_node`.
  const std::vector<int>& num_nodes_per_block() const {
    return num_nodes_per_block_;
  }
  // The number of edges for each basic block in the batch. Corresponds to
  // `GraphsTuple.n_edge`.
  const std::vector<int>& num_edges_per_block() const {
    return num_edges_per_block_;
  }

  // The types of the nodes in the batch.
  const std::vector<NodeType>& node_types() const { return node_types_; }
  // Feature value of the nodes in the batch (i.e. the indices of the tokens
  // corresponding to the nodes). Corresponds to `GraphsTuple.nodes`.
  const std::vector<int>& node_features() const { return node_features_; }

  // Names of types of instruction annotations stored.
  const std::set<std::string>& annotation_names() const {
    return annotation_names_;
  }
  // Values of instruction level runtime annotations. Represents a
  // `num_instructions` x `annotation_names.size()` matrix, each entry of which
  // represents the value of the annotation of the type corresponding to the
  // column for the instruction corresponding to the row.
  const std::vector<std::vector<double>>& instruction_annotations() const {
    return instruction_annotations_;
  }

  // The sender (start) nodes of the edges in the graphs. `edge_senders()[i]` is
  // the index of the start node of the i-th edge in the graph. Corresponds to
  // `GraphsTuple.senders`.
  const std::vector<NodeIndex>& edge_senders() const { return edge_senders_; }
  // The receiver (end) nodes of the edges in the graph. `edge_receivers()[i]`is
  // the index of the end node of the i-th edge in the graph. Corresponds to
  // `GraphsTuple.receivers`.
  const std::vector<NodeIndex>& edge_receivers() const {
    return edge_receivers_;
  }
  // The types of the edges in the graph. `edge_types()[i]` is the type of the
  // i-th edge in the graph.
  const std::vector<EdgeType>& edge_types() const { return edge_types_; }

  // Returns the matrix of global features of the graphs in the batch. This is a
  // 2D matrix of shape (num_nodes(), num_node_tokens()), in the row-major
  // format. Corresponds to `GraphsTuple.globals`.
  const std::vector<std::vector<int>>& global_features() const {
    return global_features_;
  }

  // Returns a vector of node features. The feature of each node is the index of
  // the edge type (i.e. the numerical constant associated with the given value
  // of EdgeType). Corresponds to `GraphsTuple.edges`.
  std::vector<int> EdgeFeatures() const;

  // Returns a vector of boolean values of size `num_nodes()`.
  // InstructionNodeMask()[i] is true if and only if node_types()[i] is
  // NodeType::kInstruction. This vector is used by the models to extract nodes
  // corresponding to instructions in the basic block.
  std::vector<bool> InstructionNodeMask() const;

  // Returns the delta block tensor. This is a 1D tensor of num_instructions
  // integers (where num_instructions is the number of instructions in all basic
  // blocks in the current batch. For each instruction, it contains the index of
  // the basic block to which the instruction belongs.
  // For example, when the current batch contains three basic blocks with 2, 4,
  // and 1 instruction, the return value is {0, 0, 1, 1, 1, 1, 2}.
  // The return value can be used as a value of
  // model_base.ModelBase._delta_block_index_tensor.
  std::vector<int> DeltaBlockIndex() const;

  // TODO(ondrasej): Consider adding methods that directly create NumPy arrays
  // from the data in this class to avoid the extra conversion.

  // Methods for accessing the indices of the special tokens in the graph
  // builder. When they return a non-negative value, this value is the index of
  // the token in the input list of tokens. A negative value means that the
  // token is not used.
  TokenIndex immediate_token() const { return immediate_token_; }
  TokenIndex fp_immediate_token() const { return fp_immediate_token_; }
  TokenIndex address_token() const { return address_token_; }
  TokenIndex memory_token() const { return memory_token_; }
  TokenIndex replacement_token() const { return replacement_token_; }

  // Converts the contents of the graph builder to a human-readable string
  // representation.
  std::string DebugString() const;

 private:
  // Keeps track of the state of the basic block graph builder, and allows
  // reverting it to a state before adding a basic block to the current batch.
  // The class is intended to be used as an RAII object - it is created at the
  // top of a scope that adds objects to the current batch. When execution
  // leaves the scope, the newly added objects are preserved only when the user
  // called AddBasicBlockTransaction::Commit(). This simplifies restoring the
  // builder to a valid state when the builder encounters an error.
  class AddBasicBlockTransaction {
   public:
    // Initializes the transaction object. Takes a snapshot of the sizes of the
    // vectors in the basic block graph builder at the moment the constructor is
    // called.
    explicit AddBasicBlockTransaction(BasicBlockGraphBuilder* graph_builder);
    // Reverts the state of the basic block graph builder when Commit() was not
    // called.
    ~AddBasicBlockTransaction();

    // Commits the newly added objects. Note that the only thing this method
    // does is that it sets is_committed_ to true; it does not make any other
    // changes to the basic block graph builder or to this object.
    void Commit();

   private:
    // Manually resets the basic block graph builder to the state before the
    // transaction object was created; resizes all the vectors in the builder to
    // their original size.
    void Rollback();

    // The basic block graph builder managed by the transaction.
    BasicBlockGraphBuilder& graph_builder_;
    // True when Commit() was called; otherwise, false.
    bool is_committed_ = false;

    // The sizes of the vectors in the basic block graph builder at the time
    // when the transaction was created.
    size_t prev_num_nodes_per_block_size_;
    size_t prev_num_edges_per_block_size_;
    size_t prev_node_types_size_;
    size_t prev_node_features_size_;
    size_t prev_edge_senders_size_;
    size_t prev_edge_receivers_size_;
    size_t prev_edge_types_size_;
    size_t prev_global_features_size_;
  };

  // Adds nodes and edges for a single input operand of an instruction.
  bool AddInputOperand(NodeIndex instruction_node,
                       const InstructionOperand& operand);
  // Adds nodes and edges for a single output operand of an instruction.
  bool AddOutputOperand(NodeIndex instruction_node,
                        const InstructionOperand& operand);

  // Adds dependency of a node (instruction or an address computation node) on
  // a register. Adds the register node if it doesn't exist in the graph.
  bool AddDependencyOnRegister(NodeIndex dependent_node,
                               const std::string& register_name,
                               EdgeType edge_type);

  // Adds a new node to the batch; the feature of the node is given directly by
  // the caller.
  NodeIndex AddNode(NodeType node_type, TokenIndex token_index);
  // Adds a new edge to the batch; the feature of the node is determined from
  // the token associated with the node. Returns kInvalidNode when the node was
  // not added.
  NodeIndex AddNode(NodeType node_type, const std::string& token);
  // Adds a new edge to the batch.
  void AddEdge(EdgeType edge_type, NodeIndex sender, NodeIndex receiver);

  // Mapping from string node tokens to indices of embedding vectors used in
  // the models.
  const std::unordered_map<std::string, TokenIndex> node_tokens_;
  // Tokens corresponding to nodes in the batch that are not associated directly
  // with a token of the assembly language.
  const TokenIndex immediate_token_;
  const TokenIndex fp_immediate_token_;
  const TokenIndex address_token_;
  const TokenIndex memory_token_;

  // Holds valid annotation names in sorted order. Instruction annotations with
  // names belonging to this list are stored in `instruction_annotations_` and
  // the rest are discarded.
  const std::set<std::string> annotation_names_;

  const OutOfVocabularyTokenBehavior out_of_vocabulary_behavior_;
  const TokenIndex replacement_token_;

  std::vector<int> num_nodes_per_block_;
  std::vector<int> num_edges_per_block_;

  std::vector<NodeType> node_types_;
  std::vector<TokenIndex> node_features_;

  // Mapping from annotation type names to corresponding row index in the
  // `instruction_annotations_` matrix.
  std::unordered_map<std::string, int> annotation_name_to_idx_;
  std::vector<std::vector<double>> instruction_annotations_;

  std::vector<NodeIndex> edge_senders_;
  std::vector<NodeIndex> edge_receivers_;
  std::vector<EdgeType> edge_types_;

  std::vector<std::vector<int>> global_features_;

  std::unordered_map<std::string_view, NodeIndex> register_nodes_;
  std::unordered_map<int, NodeIndex> alias_group_nodes_;
};

}  // namespace gematria

#endif  // GEMATRIA_GRANITE_GRAPH_BUILDER_H_
