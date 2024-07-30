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

#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "gematria/basic_block/basic_block_protos.h"
#include "gematria/datasets/bhive_importer.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/llvm/llvm_to_absl.h"
#include "gematria/utils/string.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "quipper/perf_parser.h"
#include "quipper/perf_reader.h"

namespace gematria {

AnnotatingImporter::AnnotatingImporter(const Canonicalizer *canonicalizer)
    : importer_(canonicalizer), perf_reader_(), perf_parser_(&perf_reader_) {
  quipper::PerfParserOptions parser_opts;
  parser_opts.do_remap = true;
  parser_opts.discard_unused_events = true;
  parser_opts.sort_events_by_time = false;
  perf_parser_.set_options(parser_opts);
}

absl::Status AnnotatingImporter::LoadPerfData(std::string_view file_name) {
  // Read and parse the `perf.data`-like file into something more tractable.
  if (!perf_reader_.ReadFile(file_name.data())) {
    return absl::InvalidArgumentError(
        "The given `perf.data`-like file could not be read.");
  }
  if (!perf_parser_.ParseRawEvents()) {
    return absl::InvalidArgumentError(
        "The given `perf.data`-like file could not be parsed.");
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<
    std::tuple<uint64_t, uint64_t, std::vector<DisassembledInstruction>>>>
AnnotatingImporter::GetBlocksFromELF(std::string_view file_name) {
  // Obtain a reference to the underlying object.
  llvm::Expected<llvm::object::OwningBinary<llvm::object::Binary>>
      owning_binary = llvm::object::createBinary(file_name);
  if (llvm::Error error = owning_binary.takeError()) {
    return LlvmErrorToStatus(std::move(error));
  }
  const llvm::object::Binary &binary = *owning_binary.get().getBinary();
  const llvm::object::ObjectFile *object =
      llvm::cast<llvm::object::ObjectFile>(&binary);

  // Make sure the object is an ELF file.
  if (!object->isELF() || !object->is64Bit() || !object->isLittleEndian()) {
    return absl::InvalidArgumentError(
        "The given object is not in ELF64LE format.");
  }
  const auto *elf_object =
      llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(object);
  if (!elf_object) {
    return absl::InvalidArgumentError(
        "Could not cast the object to an ELF64LEObjectFile.");
  }

  // Find file offset to adjust block addresses to start from zero.
  uint64_t file_offset = 0;
  auto program_headers = elf_object->getELFFile().program_headers();
  if (llvm::Error error = program_headers.takeError()) {
    return LlvmErrorToStatus(std::move(error));
  }
  for (const auto &program_header : *program_headers) {
    if (program_header.p_type == llvm::ELF::PT_LOAD &&
        program_header.p_flags & llvm::ELF::PF_R &&
        program_header.p_flags & llvm::ELF::PF_X) {
      file_offset = program_header.p_vaddr;
    }
  }

  // Read the associated `BBAddrMap` and `PGOAnalysisData`.
  std::vector<llvm::object::PGOAnalysisMap> pgo_analyses;
  llvm::Expected<std::vector<llvm::object::BBAddrMap>> bb_addr_map =
      elf_object->readBBAddrMap(
          /* TextSectionIndex = */ std::nullopt,
          /* PGOAnalyses = */ &pgo_analyses);
  if (llvm::Error error = bb_addr_map.takeError()) {
    return LlvmErrorToStatus(std::move(error));
  }

  // Populate a vector with all of the basic blocks and their addresses.
  std::vector<
      std::tuple<uint64_t, uint64_t, std::vector<DisassembledInstruction>>>
      basic_blocks;
  llvm::StringRef binary_buf = binary.getData();
  for (const llvm::object::BBAddrMap &map : bb_addr_map.get()) {
    uint64_t function_addr = map.getFunctionAddress();
    for (const llvm::object::BBAddrMap::BBRangeEntry &bb_range :
         map.getBBRanges()) {
      for (const llvm::object::BBAddrMap::BBEntry &bb : bb_range.BBEntries) {
        uint64_t begin_idx = function_addr + bb.Offset,
                 end_idx = begin_idx + bb.Size;
        llvm::ArrayRef<uint8_t> machine_code(
            reinterpret_cast<const uint8_t *>(
                binary_buf.slice(begin_idx, end_idx).data()),
            end_idx - begin_idx);
        absl::StatusOr<std::vector<DisassembledInstruction>> instructions =
            importer_.DisassembledInstructionsFromMachineCode(
                machine_code, begin_idx - file_offset);
        if (!instructions.ok()) {
          return instructions.status();
        }
        // TODO(vbshah): Consider removing the addresses and just using the
        // `addr` fields on the instructions.
        basic_blocks.emplace_back(begin_idx - file_offset,
                                  end_idx - file_offset, instructions.value());
      }
    }
  }

  return basic_blocks;
}

absl::StatusOr<std::pair<std::vector<std::string>,
                         std::unordered_map<uint64_t, std::vector<int>>>>
AnnotatingImporter::GetSamples() {
  const quipper::PerfDataProto &perf_data_proto = perf_reader_.proto();

  // Extract event type information.
  std::vector<std::string> sample_types(perf_data_proto.event_types_size());
  std::unordered_map<int, int> event_code_to_idx;
  for (int sample_type_idx = 0; sample_type_idx < sample_types.size();
       ++sample_type_idx) {
    const auto &event_type = perf_data_proto.event_types()[sample_type_idx];
    sample_types[sample_type_idx] = event_type.name();
    event_code_to_idx[event_type.id()] = sample_type_idx;
  }
  std::unordered_map<int, int> event_id_to_code;
  for (const auto &event_type : perf_data_proto.file_attrs()) {
    // Mask out bits identifying the PMU and not the event.
    int event_code = event_type.attr().config() & 0xffff;
    for (int event_id : event_type.ids()) {
      event_id_to_code[event_id] = event_code;
    }
  }

  // Find the relevant mapping.
  quipper::PerfDataProto::MMapEvent mmap;
  for (const auto &event : perf_data_proto.events()) {
    // TODO(virajbshah): Not sure if this always works, i.e. does the main
    // binary always correspond to the first MMapEvent. Implement BuildID or
    // name based checking.
    if (event.has_mmap_event() &&
        event.mmap_event().prot() & 1 /* PROT_READ */ &&
        event.mmap_event().prot() & 4 /* PROT_EXEC */) {
      mmap = event.mmap_event();
      break;
    }
  }
  // TODO(virajbshah): Make sure the mapping was found. (Use num_mmap_events)

  // If the profile has multiple event types, lookups are needed to find the
  // event type corresponding to a sample. In the other case, this is neither
  // required nor possible - since samples are not associated with IDs for
  // lookup in the proto.
  const bool has_multiple_sample_types = sample_types.size() > 1;

  // Process sample events.
  std::unordered_map<uint64_t, std::vector<int>> samples;
  for (const auto &event : perf_data_proto.events()) {
    // Filter out non-sample events.
    if (!event.has_sample_event()) {
      continue;
    }

    // Filter out sample events from outside the profiled binary.
    uint64_t sample_ip = event.sample_event().ip();
    if (!(mmap.start() <= sample_ip && sample_ip < mmap.start() + mmap.len())) {
      continue;
    }

    if (!samples.count(sample_ip)) {
      samples[sample_ip] = std::vector<int>(sample_types.size(), 0);
    }
    int event_idx = 0;
    if (has_multiple_sample_types) {
      event_idx =
          event_code_to_idx[event_id_to_code[event.sample_event().id()]];
    }
    samples[sample_ip][event_idx] += 1;
  }

  return make_pair(sample_types, samples);
}

absl::StatusOr<
    std::unordered_map<uint64_t, std::pair<uint64_t, std::vector<uint32_t>>>>
AnnotatingImporter::GetLBRData() {
  const quipper::PerfDataProto &perf_data_proto = perf_reader_.proto();

  std::unordered_map<uint64_t, std::pair<uint64_t, std::vector<uint32_t>>>
      lbr_data;
  for (const auto &event : perf_data_proto.events()) {
    if (!event.has_sample_event() ||
        !event.sample_event().branch_stack_size()) {
      continue;
    }
    const auto &branch_stack = event.sample_event().branch_stack();
    for (int branch_idx = 0; branch_idx + 1 < branch_stack.size();
         ++branch_idx) {
      const auto &next_branch_entry = branch_stack.at(branch_idx);
      const auto &branch_entry = branch_stack.at(branch_idx + 1);
      if (lbr_data.count(branch_entry.to_ip())) {
        auto &[run_end, cycles_values] = lbr_data[branch_entry.to_ip()];
        if (next_branch_entry.from_ip() == run_end) {
          cycles_values.push_back(next_branch_entry.cycles());
        } else if (next_branch_entry.from_ip() < run_end) {
          run_end = next_branch_entry.from_ip();
          cycles_values = std::vector<uint32_t>{next_branch_entry.cycles()};
        }
      } else {
        lbr_data[branch_entry.to_ip()] =
            std::make_pair(next_branch_entry.from_ip(),
                           std::vector<uint32_t>{next_branch_entry.cycles()});
      }
    }
  }

  return lbr_data;
}

absl::StatusOr<std::vector<BasicBlockWithThroughputProto>>
AnnotatingImporter::GetAnnotatedBasicBlockProtos(
    std::string_view elf_file_name, std::string_view perf_data_file_name,
    std::string_view source_name) {
  absl::Status status = LoadPerfData(perf_data_file_name);
  if (!status.ok()) {
    return status;
  }

  // Get the raw basic blocks, perf samples, and LBR data for annotation.
  absl::StatusOr<std::vector<
      std::tuple<uint64_t, uint64_t, std::vector<DisassembledInstruction>>>>
      basic_blocks = GetBlocksFromELF(elf_file_name);
  if (!basic_blocks.ok()) {
    return basic_blocks.status();
  }
  const auto sample_types_and_samples = GetSamples();
  if (!sample_types_and_samples.ok()) {
    return sample_types_and_samples.status();
  }
  const auto &[sample_types, samples] = sample_types_and_samples.value();
  const auto &lbr_data = GetLBRData();
  if (!lbr_data.ok()) {
    return lbr_data.status();
  }

  // Convert the raw basic blocks into protos and annotate them using samples.
  std::vector<BasicBlockWithThroughputProto> basic_block_protos(
      basic_blocks->size());
  for (int block_idx = 0; block_idx < basic_blocks->size(); ++block_idx) {
    // Create the un-annotated basic block proto.
    const auto &[block_begin_addr, block_end_addr, instructions] =
        basic_blocks->at(block_idx);
    auto &basic_block_proto = basic_block_protos[block_idx];
    *basic_block_proto.mutable_basic_block() =
        importer_.BasicBlockProtoFromInstructions(instructions);

    // Loop over and annotate individual instructions.
    for (int instruction_idx = 0; instruction_idx < instructions.size();
         ++instruction_idx) {
      uint64_t instruction_addr = basic_block_proto.basic_block()
                                      .machine_instructions()[instruction_idx]
                                      .address();
      if (!samples.count(instruction_addr)) {
        continue;
      }
      const std::vector<int> &annotations = samples.at(instruction_addr);

      auto &instruction_proto = basic_block_proto.mutable_basic_block()
                                    ->mutable_canonicalized_instructions()
                                    ->at(instruction_idx);
      for (int annotation_idx = 0; annotation_idx < annotations.size();
           ++annotation_idx) {
        if (annotations[annotation_idx]) {
          *instruction_proto.add_instruction_annotations() =
              ProtoFromAnnotation(Annotation(
                  /* name = */ sample_types.at(annotation_idx),
                  /* value = */ annotations.at(annotation_idx)));
        }
      }
    }

    ThroughputWithSourceProto &throughput =
        *basic_block_protos[block_idx].add_inverse_throughputs();
    throughput.set_source(source_name);
    double throughput_value = -1;
    if (lbr_data.value().count(block_begin_addr)) {
      const auto &[run_end_addr, cycles_values] =
          lbr_data.value().at(block_begin_addr);
      if (block_end_addr == run_end_addr) {
        throughput_value =
            std::accumulate(cycles_values.begin(), cycles_values.end(), 0.0) /
            cycles_values.size();
      }
    }
    throughput.add_inverse_throughput_cycles(throughput_value);
  }

  return basic_block_protos;
}

}  // namespace gematria
