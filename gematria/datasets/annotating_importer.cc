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
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "gematria/basic_block/basic_block.h"
#include "gematria/basic_block/basic_block_protos.h"
#include "gematria/datasets/bhive_importer.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/llvm/llvm_to_absl.h"
#include "gematria/proto/throughput.pb.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "quipper/perf_data.pb.h"
#include "quipper/perf_parser.h"
#include "quipper/perf_reader.h"

namespace gematria {

// Memory mapping protection flag bits on Linux, from `sys/mman.h`.
constexpr int kProtRead = 0b001;  /* PROT_READ */
constexpr int kProtWrite = 0b010; /* PROT_WRITE */
constexpr int kProtExec = 0b100;  /* PROT_EXEC */

AnnotatingImporter::AnnotatingImporter(const Canonicalizer *canonicalizer)
    : importer_(canonicalizer) {}

absl::StatusOr<const quipper::PerfDataProto *> AnnotatingImporter::LoadPerfData(
    std::string_view file_name) {
  // Read and parse the `perf.data`-like file into something more tractable.
  if (!perf_reader_.ReadFile(std::string(file_name))) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "The given `perf.data`-like file (%s) could not be read.", file_name));
  }

  quipper::PerfParser perf_parser(
      &perf_reader_, quipper::PerfParserOptions{.do_remap = true,
                                                .discard_unused_events = true,
                                                .sort_events_by_time = false,
                                                .combine_mappings = true});
  if (!perf_parser.ParseRawEvents()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "The given `perf.data`-like file (%s) could not be parsed.",
        file_name));
  }

  return &perf_reader_.proto();
}

namespace {

llvm::StringRef GetBasenameFromPath(const llvm::StringRef path) {
  int idx = path.find_last_of('/');
  if (idx == llvm::StringRef::npos) {
    return path;
  }
  return path.substr(idx + 1);
}

}  // namespace

absl::StatusOr<const quipper::PerfDataProto_MMapEvent *>
AnnotatingImporter::GetMainMapping(
    const llvm::object::ELFObjectFileBase *elf_object,
    const quipper::PerfDataProto *perf_data) {
  llvm::StringRef file_name =
      GetBasenameFromPath(elf_object->getFileName().str());
  // TODO(vbshah): There may be multiple mappings corresponding to the profiled
  // binary. Record and match samples from all of them instead of assuming
  // there is only one and returning after finding it.
  for (const auto &event : perf_data->events()) {
    if (event.has_mmap_event() &&
        GetBasenameFromPath(event.mmap_event().filename()) == file_name &&
        event.mmap_event().prot() & kProtRead &&
        event.mmap_event().prot() & kProtExec) {
      return &event.mmap_event();
    }
  }

  return absl::InvalidArgumentError(absl::StrFormat(
      "The given `perf.data`-like file does not have a mapping corresponding"
      " to the given object (%s).",
      elf_object->getFileName()));
}

absl::StatusOr<llvm::object::OwningBinary<llvm::object::Binary>>
AnnotatingImporter::LoadBinary(std::string_view file_name) {
  // Obtain a reference to the underlying object.
  llvm::Expected<llvm::object::OwningBinary<llvm::object::Binary>>
      owning_binary = llvm::object::createBinary(file_name);
  if (llvm::Error error = owning_binary.takeError()) {
    return LlvmErrorToStatus(std::move(error));
  }

  return std::move(*owning_binary);
}

absl::StatusOr<llvm::object::ELFObjectFileBase const *>
AnnotatingImporter::GetELFFromBinary(const llvm::object::Binary *binary) {
  if (!binary->isObject()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("The given binary (%s) is not an object.",
                        std::string(binary->getFileName())));
  }
  const auto *object = llvm::cast<llvm::object::ObjectFile>(binary);
  if (object == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Could not cast the binary (%s) to an ObjectFile.",
                        std::string(binary->getFileName())));
  }

  // Make sure the object is an ELF file.
  if (!object->isELF()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("The given object (%s) is not in ELF format.",
                        std::string(binary->getFileName())));
  }
  const auto *elf_object =
      llvm::dyn_cast<llvm::object::ELFObjectFileBase>(object);
  if (elf_object == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Could not cast the object (%s) to an ELFObjectFileBase.",
        std::string(binary->getFileName())));
  }

  return elf_object;
}

absl::StatusOr<std::vector<DisassembledInstruction>>
AnnotatingImporter::GetELFSlice(
    const llvm::object::ELFObjectFileBase *elf_object, uint64_t range_begin,
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
AnnotatingImporter::GetBlocksFromELF(
    const llvm::object::ELFObjectFileBase *elf_object) {
  // Read the associated `BBAddrMap` and `PGOAnalysisData`.
  std::vector<llvm::object::PGOAnalysisMap> pgo_analyses;
  llvm::Expected<std::vector<llvm::object::BBAddrMap>> bb_addr_map =
      elf_object->readBBAddrMap(
          /* TextSectionIndex = */ std::nullopt,
          /* PGOAnalyses = */ &pgo_analyses);
  if (llvm::Error error = bb_addr_map.takeError()) {
    return LlvmErrorToStatus(std::move(error));
  }

  // TODO(vbshah): Consider making it possible to use other ELFTs rather than
  // only ELF64LE since only the implementation of GetMainProgramHeader differs
  // between different ELFTs.
  if (!elf_object->is64Bit() || !elf_object->isLittleEndian()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("The given object (%s) is not in ELF64LE format.",
                        elf_object->getFileName()));
  }
  const auto *typed_elf_object =
      llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(elf_object);
  if (typed_elf_object == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Could not cast the ELF object (%s) to an ELF64LEObjectFileBase.",
        elf_object->getFileName()));
  }

  const auto main_header = GetMainProgramHeader(typed_elf_object);
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
        uint64_t begin_idx = function_addr + bb.Offset;
        uint64_t end_idx = begin_idx + bb.Size;
        if (begin_idx == end_idx) {
          continue;  // Skip any empty basic blocks.
        }
        const auto &basic_block =
            GetELFSlice(elf_object, begin_idx, end_idx, main_header->p_vaddr);
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
AnnotatingImporter::GetSamples(
    const quipper::PerfDataProto *perf_data,
    const quipper::PerfDataProto_MMapEvent *mapping) {
  const uint64_t mmap_begin_addr = mapping->start();
  const uint64_t mmap_end_addr = mmap_begin_addr + mapping->len();

  // Extract event type information,
  const int num_sample_types = perf_data->event_types_size();
  std::vector<std::string> sample_types(num_sample_types);
  std::unordered_map<int, int> event_code_to_idx;
  for (int sample_type_idx = 0; sample_type_idx < num_sample_types;
       ++sample_type_idx) {
    const auto &event_type = perf_data->event_types()[sample_type_idx];
    sample_types[sample_type_idx] = event_type.name();
    event_code_to_idx[event_type.id()] = sample_type_idx;
  }
  std::unordered_map<int, int> event_id_to_code;
  for (const auto &event_type : perf_data->file_attrs()) {
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
  for (const auto &event : perf_data->events()) {
    // Filter out non-sample events.
    if (!event.has_sample_event()) {
      continue;
    }

    // Filter out sample events from outside the profiled binary.
    if (!event.sample_event().has_pid() ||
        !(event.sample_event().pid() == mapping->pid())) {
      continue;
    }
    uint64_t sample_ip = event.sample_event().ip();
    if (sample_ip < mmap_begin_addr || sample_ip >= mmap_end_addr) {
      continue;
    }

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
AnnotatingImporter::GetLBRBlocksWithLatency(
    const llvm::object::ELFObjectFileBase *elf_object,
    const quipper::PerfDataProto *perf_data,
    const quipper::PerfDataProto_MMapEvent *mapping) {
  // TODO(vbshah): Refactor this and other parameters as function arguments.
  constexpr int kMaxBlockSizeBytes = 65536;

  const uint64_t mmap_begin_addr = mapping->start();
  const uint64_t mmap_end_addr = mmap_begin_addr + mapping->len();

  // TODO(vbshah): Consider making it possible to use other ELFTs rather than
  // only ELF64LE since only the implementation of GetMainProgramHeader differs
  // between different ELFTs.
  if (!elf_object->is64Bit() || !elf_object->isLittleEndian()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("The given object (%s) is not in ELF64LE format.",
                        elf_object->getFileName()));
  }
  const auto *typed_elf_object =
      llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(elf_object);
  if (typed_elf_object == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Could not cast the ELF object (%s) to an ELF64LEObjectFileBase.",
        elf_object->getFileName()));
  }

  const auto main_header = GetMainProgramHeader(typed_elf_object);
  if (!main_header.ok()) {
    return main_header.status();
  }

  std::vector<
      std::pair<std::vector<DisassembledInstruction>, std::vector<uint32_t>>>
      blocks;
  std::unordered_map<std::pair<uint64_t, uint64_t>, int,
                     absl::Hash<std::pair<uint64_t, uint64_t>>>
      index_map;
  for (const auto &event : perf_data->events()) {
    if (!event.has_sample_event() ||
        !event.sample_event().branch_stack_size()) {
      continue;
    }

    // Check if the sample PID matches that of the relevant mapping.
    if (!event.sample_event().has_pid() ||
        !(event.sample_event().pid() == mapping->pid())) {
      continue;
    }

    const auto &branch_stack = event.sample_event().branch_stack();
    for (int branch_idx = branch_stack.size() - 2; branch_idx >= 0;
         --branch_idx) {
      const auto &branch_entry = branch_stack[branch_idx + 1];
      const auto &next_branch_entry = branch_stack[branch_idx];

      const uint64_t block_begin = branch_entry.to_ip();
      const uint64_t block_end = next_branch_entry.from_ip();

      // Simple validity checks: the block must start before it ends and cannot
      // be larger than some threshold.
      if (block_begin >= block_end) {
        continue;
      }
      if (block_end - block_begin > kMaxBlockSizeBytes) {
        continue;
      }

      // Remove blocks not belonging to the binary we are importing from.
      if (block_begin < mmap_begin_addr || mmap_end_addr < block_end) {
        continue;
      }

      uint32_t block_latency = next_branch_entry.cycles();

      std::pair<uint64_t, uint64_t> block_range = {block_begin, block_end};
      if (index_map.count(block_range)) {
        blocks[index_map[block_range]].second.push_back(block_latency);
      } else {
        index_map[block_range] = blocks.size();
        const auto block = GetELFSlice(elf_object, block_begin, block_end,
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
  // Try to load the binary and cast it down to an ELF object.
  absl::StatusOr<llvm::object::OwningBinary<llvm::object::Binary>>
      owning_binary = LoadBinary(elf_file_name);
  if (!owning_binary.ok()) {
    return owning_binary.status();
  }
  const auto elf_object = GetELFFromBinary(owning_binary->getBinary());
  if (!elf_object.ok()) {
    return elf_object.status();
  }

  // Try to load the perf profile and locate its main mapping, i.e. the one
  // corresponding to the executable load segment of the given object file.
  absl::StatusOr<const quipper::PerfDataProto *> perf_data =
      LoadPerfData(perf_data_file_name);
  if (!perf_data.ok()) {
    return perf_data.status();
  }
  auto main_mapping = GetMainMapping(*elf_object, *perf_data);
  if (!main_mapping.ok()) {
    return main_mapping.status();
  }

  // Get the raw basic blocks, perf samples, and LBR data for annotation.
  absl::StatusOr<std::vector<
      std::pair<std::vector<DisassembledInstruction>, std::vector<uint32_t>>>>
      basic_blocks =
          GetLBRBlocksWithLatency(*elf_object, *perf_data, *main_mapping);
  if (!basic_blocks.ok()) {
    return basic_blocks.status();
  }
  const auto sample_types_and_samples = GetSamples(*perf_data, *main_mapping);
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
    throughput.add_inverse_throughput_cycles(
        std::accumulate(latency.begin(), latency.end(), 0.0) / latency.size());
  }

  return basic_block_protos;
}

}  // namespace gematria
