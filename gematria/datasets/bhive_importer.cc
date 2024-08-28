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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "gematria/basic_block/basic_block_protos.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/llvm/llvm_to_absl.h"
#include "gematria/proto/basic_block.pb.h"
#include "gematria/proto/throughput.pb.h"
#include "gematria/utils/string.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"

namespace gematria {
namespace {

// The assembly syntax used in the output protos. The assembly code in the proto
// is purely informative, so we simply use the "default" assembly syntax for
// each architecture, since it is guaranteed to always be there.
constexpr int kDefaultSyntax = 0;

}  // namespace

BHiveImporter::BHiveImporter(const Canonicalizer* canonicalizer)
    : canonicalizer_(*ABSL_DIE_IF_NULL(canonicalizer)),
      target_machine_(canonicalizer->target_machine()),
      context_(std::make_unique<llvm::MCContext>(
          target_machine_.getTargetTriple(), target_machine_.getMCAsmInfo(),
          target_machine_.getMCRegisterInfo(),
          target_machine_.getMCSubtargetInfo())),
      disassembler_(target_machine_.getTarget().createMCDisassembler(
          *target_machine_.getMCSubtargetInfo(), *context_)),
      mc_inst_printer_(target_machine_.getTarget().createMCInstPrinter(
          target_machine_.getTargetTriple(), kDefaultSyntax,
          *target_machine_.getMCAsmInfo(), *target_machine_.getMCInstrInfo(),
          *target_machine_.getMCRegisterInfo())) {}

BasicBlockProto BHiveImporter::BasicBlockProtoFromInstructions(
    llvm::ArrayRef<DisassembledInstruction> disassembled_instructions,
    uint64_t base_address /*= 0*/) {
  BasicBlockProto basic_block_proto;
  for (const DisassembledInstruction& instruction : disassembled_instructions) {
    MachineInstructionProto& machine_instruction =
        *basic_block_proto.add_machine_instructions();
    machine_instruction.set_address(instruction.address);
    machine_instruction.set_assembly(instruction.assembly);
    machine_instruction.set_machine_code(instruction.machine_code);
    *basic_block_proto.add_canonicalized_instructions() = ProtoFromInstruction(
        canonicalizer_.InstructionFromMCInst(instruction.mc_inst));
  }
  return basic_block_proto;
}

absl::StatusOr<std::vector<DisassembledInstruction>>
BHiveImporter::DisassembledInstructionsFromMachineCode(
    llvm::ArrayRef<uint8_t> machine_code, uint64_t base_address /* = 0 */) {
  llvm::Expected<std::vector<DisassembledInstruction>> instructions =
      DisassembleAllInstructions(*disassembler_,
                                 *target_machine_.getMCInstrInfo(),
                                 *target_machine_.getMCRegisterInfo(),
                                 *target_machine_.getMCSubtargetInfo(),
                                 *mc_inst_printer_, base_address, machine_code);
  if (llvm::Error error = instructions.takeError()) {
    return LlvmErrorToStatus(std::move(error));
  }

  return *instructions;
}

absl::StatusOr<BasicBlockProto> BHiveImporter::BasicBlockProtoFromMachineCode(
    llvm::ArrayRef<uint8_t> machine_code, uint64_t base_address /*= 0*/) {
  auto instructions =
      DisassembledInstructionsFromMachineCode(machine_code, base_address);
  if (!instructions.ok()) {
    return instructions.status();
  }

  return BasicBlockProtoFromInstructions(instructions.value());
}

absl::StatusOr<BasicBlockProto>
BHiveImporter::BasicBlockProtoFromMachineCodeHex(
    std::string_view machine_code_hex, uint64_t base_address /*= 0*/) {
  const auto machine_code_bytes_or_status = ParseHexString(machine_code_hex);
  if (!machine_code_bytes_or_status.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("cannot parse: ", machine_code_hex));
  }

  return BasicBlockProtoFromMachineCode(machine_code_bytes_or_status.value(),
                                        base_address);
}

absl::StatusOr<BasicBlockWithThroughputProto> BHiveImporter::ParseBHiveCsvLine(
    std::string_view source_name, std::string_view line,
    size_t machine_code_hex_column_index, size_t throughput_column_index,
    double throughput_scaling /*= 1.0*/, uint64_t base_address /*= 0*/) {
  const absl::InlinedVector<std::string_view, 2> columns =
      absl::StrSplit(line, ',');
  const int min_required_num_columns =
      std::max(machine_code_hex_column_index, throughput_column_index) + 1;
  if (columns.size() < min_required_num_columns) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected `line` to have at least %d columns, found %d: %s",
        min_required_num_columns, columns.size(), line));
  }
  if (machine_code_hex_column_index == throughput_column_index) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected machine code column and throughput column indices to be "
        "different, but were both %d: %s",
        machine_code_hex_column_index, line));
  }
  const std::string_view machine_code_hex =
      columns[machine_code_hex_column_index];
  const std::string_view throughput_str = columns[throughput_column_index];

  BasicBlockWithThroughputProto proto;
  absl::StatusOr<BasicBlockProto> block_proto_or_status =
      BasicBlockProtoFromMachineCodeHex(machine_code_hex, base_address);
  if (!block_proto_or_status.ok()) return block_proto_or_status.status();
  *proto.mutable_basic_block() = std::move(block_proto_or_status).value();

  double throughput_cycles = 0.0;
  if (!absl::SimpleAtod(throughput_str, &throughput_cycles)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not parse throughput value ", throughput_str));
  }

  ThroughputWithSourceProto& throughput = *proto.add_inverse_throughputs();
  throughput.set_source(source_name);
  throughput.add_inverse_throughput_cycles(throughput_cycles *
                                           throughput_scaling);

  return proto;
}

}  // namespace gematria
