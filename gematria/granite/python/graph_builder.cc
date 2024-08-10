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

#include <set>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "gematria/model/oov_token_behavior.h"
#include "gematria/proto/canonicalized_instruction.pb.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11_protobuf/native_proto_caster.h"

namespace gematria {
namespace {

namespace py = ::pybind11;

constexpr const char* const kModuleDocstring =
    R"(Conversion of basic blocks to a graph representation.

See the comments in the C++ version of the class for more details on the graph
representation and the conversion process.)";

PYBIND11_MODULE(graph_builder, m) {
  m.doc() = kModuleDocstring;

  pybind11_protobuf::ImportNativeProtoCasters();

  py::enum_<NodeType>(m, "NodeType")
      .value("INSTRUCTION", NodeType::kInstruction)
      .value("REGISTER", NodeType::kRegister)
      .value("IMMEDIATE", NodeType::kImmediate)
      .value("FP_IMMEDIATE", NodeType::kFpImmediate)
      .value("ADDRESS_OPERAND", NodeType::kAddressOperand)
      .value("MEMORY_OPERAND", NodeType::kMemoryOperand)
      .export_values();

  py::enum_<EdgeType>(m, "EdgeType")
      .value("STRUCTURAL_DEPENDENCY", EdgeType::kStructuralDependency)
      .value("REVERSE_STRUCTURAL_DEPENDENCY",
             EdgeType::kReverseStructuralDependency)
      .value("INPUT_OPERANDS", EdgeType::kInputOperands)
      .value("OUTPUT_OPERANDS", EdgeType::kOutputOperands)
      .value("ADDRESS_BASE_REGISTER", EdgeType::kAddressBaseRegister)
      .value("ADDRESS_INDEX_REGISTER", EdgeType::kAddressIndexRegister)
      .value("ADDRESS_SEGMENT_REGISTER", EdgeType::kAddressSegmentRegister)
      .value("ADDRESS_DISPLACEMENT", EdgeType::kAddressDisplacement)
      .value("INSTRUCTION_PREFIX", EdgeType::kInstructionPrefix)
      .export_values();

  py::class_<BasicBlockGraphBuilder>(m, "BasicBlockGraphBuilder")
      .def(
          py::init<std::vector<std::string> /* node_tokens */,
                   absl::string_view /* immediate_token */,
                   absl::string_view /* fp_immediate_token */,
                   absl::string_view /* address_token */,
                   absl::string_view /* memory_token */,
                   std::vector<std::string> /* annotation_names */,
                   OutOfVocabularyTokenBehavior /* out_of_vocabulary_behavior */
                   >(),
          py::arg("node_tokens"), py::arg("immediate_token"),
          py::arg("fp_immediate_token"), py::arg("address_token"),
          py::arg("memory_token"),
          py::arg("annotation_names") = std::vector<std::string>(),
          py::arg("out_of_vocabulary_behavior"))
      .def("add_basic_block", &BasicBlockGraphBuilder::AddBasicBlock,
           py::arg("block"))
      .def("add_basic_block_from_instructions",
           &BasicBlockGraphBuilder::AddBasicBlockFromInstructions,
           py::arg("instructions"))
      .def("reset", &BasicBlockGraphBuilder::Reset)
      .def_property_readonly("num_node_tokens",
                             &BasicBlockGraphBuilder::num_node_tokens)
      .def_property_readonly("num_graphs", &BasicBlockGraphBuilder::num_graphs)
      .def_property_readonly("num_nodes", &BasicBlockGraphBuilder::num_nodes)
      .def_property_readonly("num_edges", &BasicBlockGraphBuilder::num_edges)
      .def_property_readonly("num_nodes_per_block",
                             &BasicBlockGraphBuilder::num_nodes_per_block)
      .def_property_readonly("num_edges_per_block",
                             &BasicBlockGraphBuilder::num_edges_per_block)
      .def_property_readonly("node_features",
                             &BasicBlockGraphBuilder::node_features)
      .def_property_readonly("instruction_node_mask",
                             &BasicBlockGraphBuilder::InstructionNodeMask)
      .def_property_readonly("annotation_names",
                             &BasicBlockGraphBuilder::annotation_names)
      .def_property_readonly("instruction_annotations",
                             &BasicBlockGraphBuilder::instruction_annotations)
      .def_property_readonly("annotation_names",
                             &BasicBlockGraphBuilder::annotation_names)
      .def_property_readonly("instruction_annotations",
                             &BasicBlockGraphBuilder::instruction_annotations)
      .def_property_readonly("edge_senders",
                             &BasicBlockGraphBuilder::edge_senders)
      .def_property_readonly("edge_receivers",
                             &BasicBlockGraphBuilder::edge_receivers)
      .def_property_readonly("edge_features",
                             &BasicBlockGraphBuilder::EdgeFeatures)
      .def_property_readonly("global_features",
                             &BasicBlockGraphBuilder::global_features)
      .def_property_readonly("immediate_token",
                             &BasicBlockGraphBuilder::immediate_token)
      .def_property_readonly("fp_immediate_token",
                             &BasicBlockGraphBuilder::fp_immediate_token)
      .def_property_readonly("address_token",
                             &BasicBlockGraphBuilder::address_token)
      .def_property_readonly("memory_token",
                             &BasicBlockGraphBuilder::memory_token)
      .def_property_readonly("replacement_token",
                             &BasicBlockGraphBuilder::replacement_token);
}

}  // namespace
}  // namespace gematria
