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
#include <tuple>
#include <unordered_map>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
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
    : importer_(canonicalizer), perf_parser_(&perf_reader_) {
  quipper::PerfParserOptions parser_opts;
  parser_opts.do_remap = true;
  parser_opts.discard_unused_events = true;
  parser_opts.sort_events_by_time = false;
  perf_parser_.set_options(parser_opts);
}

absl::Status AnnotatingImporter::LoadPerfData(std::string_view file_name) {
  // Read and parse the `perf.data`-like file into something more tractable.
  if (!perf_reader_.ReadFile(std::string(file_name))) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "The given `perf.data`-like file (%s) could not be read.", file_name));
  }
  if (!perf_parser_.ParseRawEvents()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "The given `perf.data`-like file (%s) could not be parsed.",
        file_name));
  }

  // Find the relevant mapping.
  // TODO(virajbshah): Make sure the mapping was found. (Use num_mmap_events)
  const quipper::PerfDataProto &perf_data_proto = perf_reader_.proto();
  for (const auto &event : perf_data_proto.events()) {
    // TODO(virajbshah): Not sure if this always works, i.e. does the main
    // binary always correspond to the first MMapEvent. Implement BuildID or
    // name based checking.
    if (event.has_mmap_event() &&
        event.mmap_event().prot() & 1 /* PROT_READ */ &&
        event.mmap_event().prot() & 4 /* PROT_EXEC */) {
      main_mapping_ = event.mmap_event();
      break;
    }
  }

  return absl::OkStatus();
}

absl::Status AnnotatingImporter::LoadBinary(std::string_view file_name) {
  // Obtain a reference to the underlying object.
  llvm::Expected<llvm::object::OwningBinary<llvm::object::Binary>>
      owning_binary = llvm::object::createBinary(file_name);
  if (llvm::Error error = owning_binary.takeError()) {
    return LlvmErrorToStatus(std::move(error));
  }
  owning_binary_ = std::move(*owning_binary);

  return absl::OkStatus();
}

absl::StatusOr<llvm::object::ELF64LEObjectFile *>
AnnotatingImporter::GetELFFromBinary() {
  llvm::object::Binary *binary = owning_binary_.getBinary();
  if (!binary->isObject()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("The given binary (%s) is not an object.",
                        std::string(binary->getFileName())));
  }
  llvm::object::ObjectFile *object =
      llvm::cast<llvm::object::ObjectFile>(binary);
  if (!object) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Could not cast the binary (%s) to an ObjectFile.",
                        std::string(binary->getFileName())));
  }

  // Make sure the object is an ELF file.
  if (!object->isELF() || !object->is64Bit() || !object->isLittleEndian()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("The given object (%s) is not in ELF64LE format.",
                        std::string(binary->getFileName())));
  }
  auto *elf_object = llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(object);
  if (!elf_object) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Could not cast the object (%s) to an ELF64LEObjectFile.",
        std::string(binary->getFileName())));
  }

  return elf_object;
}

absl::StatusOr<llvm::object::Elf_Phdr_Impl<llvm::object::ELF64LE>>
AnnotatingImporter::GetMainProgramHeader(
    const llvm::object::ELF64LEObjectFile *elf_object) {
  llvm::object::Elf_Phdr_Impl<llvm::object::ELF64LE> main_header;
  bool found_main_header = false;
  auto program_headers = elf_object->getELFFile().program_headers();
  if (llvm::Error error = program_headers.takeError()) {
    return LlvmErrorToStatus(std::move(error));
  }
  for (const auto &program_header : *program_headers) {
    if (program_header.p_type == llvm::ELF::PT_LOAD &&
        program_header.p_flags & llvm::ELF::PF_R &&
        program_header.p_flags & llvm::ELF::PF_X) {
      if (found_main_header) {
        return absl::InvalidArgumentError(
            "The given object has multiple executable segments. This is "
            "currently not supported.");
      }
      main_header = program_header;
      found_main_header = true;
    }
  }

  return main_header;
}

absl::StatusOr<std::vector<DisassembledInstruction>>
AnnotatingImporter::GetELFSlice(
    const llvm::object::ELF64LEObjectFile *elf_object, uint64_t range_begin,
    uint64_t range_end, uint64_t file_offset) {
  llvm::StringRef binary_buf = elf_object->getData();

  llvm::ArrayRef<uint8_t> machine_code(
      reinterpret_cast<const uint8_t *>(
          binary_buf.slice(range_begin, range_end).data()),
      range_end - range_begin);
  absl::StatusOr<std::vector<DisassembledInstruction>> instructions =
      importer_.DisassembledInstructionsFromMachineCode(
          machine_code, range_begin - file_offset);
  if (!instructions.ok()) {
    return instructions.status();
  }
  return instructions;
}

absl::StatusOr<std::vector<std::vector<DisassembledInstruction>>>
AnnotatingImporter::GetBlocksFromELF() {
  const auto elf_object = GetELFFromBinary();
  if (!elf_object.ok()) {
    return elf_object.status();
  }

  // Read the associated `BBAddrMap` and `PGOAnalysisData`.
  std::vector<llvm::object::PGOAnalysisMap> pgo_analyses;
  llvm::Expected<std::vector<llvm::object::BBAddrMap>> bb_addr_map =
      (*elf_object)
          ->readBBAddrMap(
              /* TextSectionIndex = */ std::nullopt,
              /* PGOAnalyses = */ &pgo_analyses);
  if (llvm::Error error = bb_addr_map.takeError()) {
    return LlvmErrorToStatus(std::move(error));
  }
  const auto main_header = GetMainProgramHeader(*elf_object);
  if (!main_header.ok()) {
    return main_header.status();
  }

  // Populate a vector with all of the basic blocks.
  std::vector<std::vector<DisassembledInstruction>> basic_blocks;
  for (const llvm::object::BBAddrMap &map : bb_addr_map.get()) {
    uint64_t function_addr = map.getFunctionAddress();
    for (const llvm::object::BBAddrMap::BBRangeEntry &bb_range :
         map.getBBRanges()) {
      for (const llvm::object::BBAddrMap::BBEntry &bb : bb_range.BBEntries) {
        uint64_t begin_idx = function_addr + bb.Offset,
                 end_idx = begin_idx + bb.Size;
        if (begin_idx == end_idx) {
          continue;  // Skip any empty basic blocks.
        }
        const auto &basic_block =
            GetELFSlice(*elf_object, begin_idx, end_idx, main_header->p_vaddr);
        if (!basic_block.ok()) {
          return basic_block.status();
        }
        basic_blocks.push_back(*basic_block);
      }
    }
  }
  return basic_blocks;
}

absl::StatusOr<std::pair<std::vector<std::string>,
                         std::unordered_map<uint64_t, std::vector<int>>>>
AnnotatingImporter::GetSamples() {
  const quipper::PerfDataProto &perf_data_proto = perf_reader_.proto();
  const uint64_t mmap_begin_addr = main_mapping_.start();
  const uint64_t mmap_end_addr = main_mapping_.start() + main_mapping_.len();

  // Extract event type information,
  const int num_sample_types = perf_data_proto.event_types_size();
  std::vector<std::string> sample_types(num_sample_types);
  std::unordered_map<int, int> event_code_to_idx;
  for (int sample_type_idx = 0; sample_type_idx < num_sample_types;
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

  // If the profile has multiple event types, lookups are needed to find the
  // event type corresponding to a sample. In the other case, this is neither
  // required nor possible - since samples are not associated with IDs for
  // lookup in the proto.
  const bool has_multiple_sample_types = num_sample_types > 1;

  // Process sample events.
  std::unordered_map<uint64_t, std::vector<int>> samples;
  for (const auto &event : perf_data_proto.events()) {
    // Filter out non-sample events.
    if (!event.has_sample_event()) {
      continue;
    }

    // Filter out sample events from outside the profiled binary.
    uint64_t sample_ip = event.sample_event().ip();
    if (sample_ip < mmap_begin_addr || sample_ip >= mmap_end_addr) continue;

    std::vector<int> &samples_at_same_addr = samples[sample_ip];
    if (samples_at_same_addr.empty()) {
      samples_at_same_addr.resize(num_sample_types);
    }
    int event_idx = 0;
    if (has_multiple_sample_types) {
      event_idx =
          event_code_to_idx[event_id_to_code[event.sample_event().id()]];
    }
    samples_at_same_addr[event_idx] += 1;
  }

  return make_pair(sample_types, samples);
}

absl::StatusOr<std::vector<
    std::pair<std::vector<DisassembledInstruction>, std::vector<uint32_t>>>>
AnnotatingImporter::GetLBRBlocksWithLatency() {
  // TODO(vbshah): Refactor this and other parameters as function arguments.
  constexpr int kMaxBlockSizeBytes = 65536;

  const quipper::PerfDataProto &perf_data_proto = perf_reader_.proto();
  const uint64_t mmap_begin_addr = main_mapping_.start();
  const uint64_t mmap_end_addr = main_mapping_.start() + main_mapping_.len();
  const auto elf_object = GetELFFromBinary();
  if (!elf_object.ok()) {
    return elf_object.status();
  }
  const auto main_header = GetMainProgramHeader(*elf_object);
  if (!main_header.ok()) {
    return main_header.status();
  }

  std::vector<
      std::pair<std::vector<DisassembledInstruction>, std::vector<uint32_t>>>
      blocks;
  std::unordered_map<std::pair<uint64_t, uint64_t>, int,
                     absl::Hash<std::pair<uint64_t, uint64_t>>>
      index_map;
  for (const auto &event : perf_data_proto.events()) {
    if (!event.has_sample_event() ||
        !event.sample_event().branch_stack_size()) {
      continue;
    }
    const auto &branch_stack = event.sample_event().branch_stack();
    for (int branch_idx = branch_stack.size() - 2; branch_idx >= 0;
         --branch_idx) {
      const auto &branch_entry = branch_stack[branch_idx + 1];
      const auto &next_branch_entry = branch_stack[branch_idx];

      uint64_t block_begin = branch_entry.to_ip(),
               block_end = next_branch_entry.from_ip();

      // Simple validity checks: the block must start before it ends and cannot
      // be larger than some threshold.
      if (block_begin >= block_end) continue;
      if (block_end - block_begin > kMaxBlockSizeBytes) continue;

      // Remove blocks not belonging to the binary we are importing from.
      if (block_begin < mmap_begin_addr || mmap_end_addr < block_end) continue;
      if (block_begin < main_header->p_offset ||
          main_header->p_offset + main_header->p_filesz < block_end)
        continue;

      uint32_t block_latency = next_branch_entry.cycles();

      std::pair<uint64_t, uint64_t> block_range = {block_begin, block_end};
      if (index_map.count(block_range)) {
        blocks[index_map[block_range]].second.push_back(block_latency);
      } else {
        index_map[block_range] = blocks.size();
        const auto block = GetELFSlice(*elf_object, block_begin, block_end,
                                       main_header->p_vaddr);
        if (!block.ok()) {
          // TODO(vbshah): Make the importer so something better than simply
          // exiting upon encountering something unexpected.
          return block.status();
        }
        blocks.emplace_back(*block, std::vector<uint32_t>{block_latency});
      }
    }
  }

  return blocks;
}

absl::StatusOr<std::vector<BasicBlockWithThroughputProto>>
AnnotatingImporter::GetAnnotatedBasicBlockProtos(
    std::string_view elf_file_name, std::string_view perf_data_file_name,
    std::string_view source_name) {
  absl::Status status = LoadBinary(elf_file_name);
  if (!status.ok()) {
    return status;
  }
  status = LoadPerfData(perf_data_file_name);
  if (!status.ok()) {
    return status;
  }

  // Get the raw basic blocks, perf samples, and LBR data for annotation.
  absl::StatusOr<std::vector<
      std::pair<std::vector<DisassembledInstruction>, std::vector<uint32_t>>>>
      basic_blocks = GetLBRBlocksWithLatency();
  if (!basic_blocks.ok()) {
    return basic_blocks.status();
  }
  const auto sample_types_and_samples = GetSamples();
  if (!sample_types_and_samples.ok()) {
    return sample_types_and_samples.status();
  }
  const auto &[sample_types, samples] = sample_types_and_samples.value();

  // Convert the raw basic blocks into protos and annotate them using samples.
  std::vector<BasicBlockWithThroughputProto> basic_block_protos(
      basic_blocks->size());
  for (int block_idx = 0; block_idx < basic_blocks->size(); ++block_idx) {
    const auto &[instructions, latency] = (*basic_blocks)[block_idx];

    // Create the un-annotated basic block proto.
    BasicBlockWithThroughputProto &basic_block_proto =
        basic_block_protos[block_idx];
    *basic_block_proto.mutable_basic_block() =
        importer_.BasicBlockProtoFromInstructions(instructions);

    // Loop over and annotate individual instructions.
    for (int instruction_idx = 0; instruction_idx < instructions.size();
         ++instruction_idx) {
      uint64_t instruction_addr = basic_block_proto.basic_block()
                                      .machine_instructions()[instruction_idx]
                                      .address();
      if (!samples.count(instruction_addr)) continue;

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
    throughput.add_inverse_throughput_cycles(
        std::accumulate(latency.begin(), latency.end(), 0.0) / latency.size());
  }

  return basic_block_protos;
}

}  // namespace gematria
