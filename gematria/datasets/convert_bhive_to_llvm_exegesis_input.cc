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

#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "gematria/datasets/bhive_importer.h"
#include "gematria/datasets/find_accessed_addrs.h"
#include "gematria/datasets/find_accessed_addrs_exegesis.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/llvm/llvm_to_absl.h"
#include "gematria/utils/string.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/tools/llvm-exegesis/lib/TargetSelect.h"

// Use the constants from the BHive paper for setting initial register and
// memory values. These constants are set to a high enough value to avoid
// underflow and accesses within the first page, but low enough to avoid
// exceeding the virtual address space ceiling in most cases.
constexpr uint64_t kInitialRegVal = 0x12345600;
constexpr uint64_t kInitialMemVal = 0x12345600;
constexpr unsigned kInitialMemValBitWidth = 64;
constexpr std::string_view kRegDefPrefix = "# LLVM-EXEGESIS-DEFREG ";
constexpr std::string_view kMemDefPrefix = "# LLVM-EXEGESIS-MEM-DEF ";
constexpr std::string_view kMemMapPrefix = "# LLVM-EXEGESIS-MEM-MAP ";
constexpr std::string_view kMemNamePrefix = "MEM";

enum class AnnotatorType { kExegesis, kFast };

constexpr std::pair<AnnotatorType, std::string_view> kAnnotatorTypeNames[] = {
    {AnnotatorType::kExegesis, "exegesis"}, {AnnotatorType::kFast, "fast"}};

bool AbslParseFlag(absl::string_view text, AnnotatorType* type,
                   std::string* error) {
  for (const auto& [annotator_type, type_string] : kAnnotatorTypeNames) {
    if (text == type_string) {
      *type = annotator_type;
      return true;
    }
  }

  *error = "unknown annotator type";
  return false;
}

std::string AbslUnparseFlag(AnnotatorType type) {
  for (const auto& [annotator_type, type_string] : kAnnotatorTypeNames) {
    if (annotator_type == type) return std::string(type_string);
  }

  __builtin_unreachable();
}

ABSL_FLAG(std::string, bhive_csv, "", "Filename of the input BHive CSV file");
ABSL_FLAG(
    std::string, asm_output_dir, "",
    "Directory containing output files that can be executed by llvm-exegesis");
ABSL_FLAG(AnnotatorType, annotator_implementation, AnnotatorType::kFast,
          "The annotator implementation to use.");
ABSL_FLAG(std::string, json_output_dir, "",
          "Directory containing JSON output files");
ABSL_FLAG(
    unsigned, blocks_per_json_file, std::numeric_limits<unsigned>::max(),
    "The number of annotated basic blocks to include in a single JSON file");
ABSL_FLAG(unsigned, max_bb_count, std::numeric_limits<unsigned>::max(),
          "The maximum number of basic blocks to process");
ABSL_FLAG(unsigned, report_progress_every, std::numeric_limits<unsigned>::max(),
          "The number of blocks after which to report progress.");

absl::StatusOr<gematria::AccessedAddrs> GetAccessedAddrs(
    absl::Span<const uint8_t> basic_block,
    gematria::ExegesisAnnotator* exegesis_annotator) {
  const AnnotatorType annotator_implementation =
      absl::GetFlag(FLAGS_annotator_implementation);
  switch (annotator_implementation) {
    case AnnotatorType::kFast:
      // This will only get the first segfault address.
      return gematria::FindAccessedAddrs(basic_block);
    case AnnotatorType::kExegesis:
      return gematria::LlvmExpectedToStatusOr(
          exegesis_annotator->findAccessedAddrs(
              llvm::ArrayRef(basic_block.begin(), basic_block.end())));
  }
  return absl::InvalidArgumentError("unknown annotator type");
}

bool WriteJsonFile(llvm::json::Array to_write, size_t json_file_number,
                   llvm::StringRef json_output_dir) {
  llvm::SmallString<40> json_output_file_path(json_output_dir);
  llvm::sys::path::append(json_output_file_path,
                          llvm::Twine(json_file_number).concat(".json"));
  std::error_code file_ec;
  llvm::raw_fd_ostream json_output_file(json_output_file_path, file_ec);

  if (file_ec) {
    std::cerr << "Failed to open output file: "
              << static_cast<std::string_view>(json_output_file_path.str())
              << "\n";
    return false;
  }

  json_output_file
      << llvm::formatv("{0:2}", llvm::json::Value(std::move(to_write))).str();
  return true;
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const std::string bhive_filename = absl::GetFlag(FLAGS_bhive_csv);
  if (bhive_filename.empty()) {
    std::cerr << "Error: --bhive_csv is required\n";
    return 1;
  }

  const std::string json_output_dir = absl::GetFlag(FLAGS_json_output_dir);
  const std::string asm_output_dir = absl::GetFlag(FLAGS_asm_output_dir);

  const unsigned blocks_per_json_file =
      absl::GetFlag(FLAGS_blocks_per_json_file);
  if (blocks_per_json_file <= 0) {
    std::cerr << "Error: --blocks_per_json_file must be greater than 1.\n";
    return 1;
  }

  const AnnotatorType annotator_implementation =
      absl::GetFlag(FLAGS_annotator_implementation);

  std::string initial_reg_val_str =
      gematria::ConvertHexToString(kInitialRegVal);
  std::string initial_mem_val_str =
      gematria::ConvertHexToString(kInitialMemVal);
  // Prefix the string with zeroes as llvm-exegesis assumes the bit width
  // of the memory value based on the number of characters in the string.
  if (kInitialMemValBitWidth > initial_mem_val_str.size() * 4)
    initial_mem_val_str =
        std::string(
            (kInitialMemValBitWidth - initial_mem_val_str.size() * 4) / 4,
            '0') +
        initial_mem_val_str;
  std::string register_defs_lines;
  const std::unique_ptr<gematria::LlvmArchitectureSupport> llvm_support =
      gematria::LlvmArchitectureSupport::X86_64();
  const llvm::MCRegisterInfo& reg_info = llvm_support->mc_register_info();

  // Iterate through all general purpose registers and vector registers
  // and add them to the register definitions.
  const auto& gr64_register_class =
      reg_info.getRegClass(llvm::X86::GR64_NOREX2RegClassID);
  for (unsigned i = 0; i < gr64_register_class.getNumRegs(); ++i) {
    if (gr64_register_class.getRegister(i) == llvm::X86::RIP) continue;
    llvm::StringRef reg_name =
        reg_info.getName(gr64_register_class.getRegister(i));
    register_defs_lines += llvm::Twine(kRegDefPrefix)
                               .concat(reg_name)
                               .concat(" ")
                               .concat(initial_reg_val_str)
                               .concat("\n")
                               .str();
  }
  const auto& vr128_register_class =
      reg_info.getRegClass(llvm::X86::VR128RegClassID);
  for (unsigned i = 0; i < vr128_register_class.getNumRegs(); ++i) {
    llvm::StringRef reg_name =
        reg_info.getName(vr128_register_class.getRegister(i));
    register_defs_lines += llvm::Twine(kRegDefPrefix)
                               .concat(reg_name)
                               .concat(" ")
                               .concat(initial_reg_val_str)
                               .concat("\n")
                               .str();
  }

  gematria::X86Canonicalizer canonicalizer(&llvm_support->target_machine());
  gematria::BHiveImporter bhive_importer(&canonicalizer);

  llvm::exegesis::InitializeX86ExegesisTarget();

  auto llvm_state_or_error = llvm::exegesis::LLVMState::Create("", "native");
  if (!llvm_state_or_error) {
    std::cerr << "Failed to create LLVMState\n";
    return 1;
  }

  std::unique_ptr<gematria::ExegesisAnnotator> exegesis_annotator = nullptr;
  if (annotator_implementation == AnnotatorType::kExegesis) {
    auto exegesis_annotator_or_error =
        gematria::ExegesisAnnotator::create(*llvm_state_or_error);
    if (!exegesis_annotator_or_error) {
      std::cerr << "Failed to create exegesis annotator\n";
      return 1;
    }
    exegesis_annotator = std::move(*exegesis_annotator_or_error);
  }

  std::ifstream bhive_csv_file(bhive_filename);
  llvm::json::Array processed_snippets;
  const unsigned max_bb_count = absl::GetFlag(FLAGS_max_bb_count);
  const unsigned report_progress_every =
      absl::GetFlag(FLAGS_report_progress_every);
  unsigned int file_counter = 0;
  for (std::string line; std::getline(bhive_csv_file, line);) {
    if (file_counter >= max_bb_count) break;

    auto comma_index = line.find(',');
    if (comma_index == std::string::npos) {
      std::cerr << "Invalid CSV file: no comma in line '" << line << "'\n";
      return 2;
    }

    std::string_view hex = std::string_view(line).substr(0, comma_index);
    // For each line, find the accessed addresses & disassemble instructions.
    auto bytes = gematria::ParseHexString(hex);
    if (!bytes.has_value()) {
      std::cerr << "could not parse: " << hex << "\n";
      return 3;
    }

    auto proto = bhive_importer.BasicBlockProtoFromMachineCode(*bytes);

    // Check for errors.
    if (!proto.ok()) {
      std::cerr << "Failed to disassemble block '" << hex << ": "
                << proto.status() << "\n";
      continue;
    }

    auto addrs = GetAccessedAddrs(*bytes, exegesis_annotator.get());

    if (!addrs.ok()) {
      std::cerr << "Failed to find addresses for block '" << hex
                << "': " << addrs.status() << "\n";
      std::cerr << "Block disassembly:\n";
      for (const auto& instr : proto->machine_instructions()) {
        std::cerr << "\t" << instr.assembly() << "\n";
      }
      continue;
    }

    if (!asm_output_dir.empty()) {
      // Create output file path.
      llvm::Twine output_file_path = llvm::Twine(asm_output_dir)
                                         .concat("/")
                                         .concat(llvm::Twine(file_counter))
                                         .concat(".test");

      // Open output file for writing.
      std::ofstream output_file(output_file_path.str());
      if (!output_file.is_open()) {
        std::cerr << "Failed to open output file: " << output_file_path.str()
                  << "\n";
        return 4;
      }

      // Write the register definition lines into the output file.
      output_file << register_defs_lines;

      // Multiple mappings can point to the same definition.
      if (addrs->accessed_blocks.size() > 0) {
        output_file << kMemDefPrefix << kMemNamePrefix << " "
                    << addrs->block_size << " " << initial_mem_val_str << "\n";
      }
      for (const auto& addr : addrs->accessed_blocks) {
        output_file << kMemMapPrefix << kMemNamePrefix << " " << std::dec
                    << addr << "\n";
      }

      // Append disassembled instructions.
      for (const auto& instr : proto->machine_instructions()) {
        output_file << instr.assembly() << "\n";
      }
    }

    if (!json_output_dir.empty()) {
      llvm::json::Object current_snippet;

      if (addrs->accessed_blocks.size() > 0) {
        llvm::json::Array memory_definitions;
        llvm::json::Object current_memory_definition;
        current_memory_definition["Name"] = llvm::json::Value(kMemNamePrefix);
        current_memory_definition["Size"] = addrs->block_size;
        current_memory_definition["Value"] = llvm::json::Value(kInitialMemVal);
        memory_definitions.push_back(std::move(current_memory_definition));
        current_snippet["MemoryDefinitions"] =
            llvm::json::Value(std::move(memory_definitions));

        llvm::json::Array memory_mappings;
        for (const uintptr_t addr : addrs->accessed_blocks) {
          llvm::json::Object current_memory_mapping;
          current_memory_mapping["Value"] = llvm::json::Value(kMemNamePrefix);
          current_memory_mapping["Address"] = addr;
          memory_mappings.push_back(std::move(current_memory_mapping));
        }
        current_snippet["MemoryMappings"] =
            llvm::json::Value(std::move(memory_mappings));
      } else {
        current_snippet["MemoryDefinitions"] = llvm::json::Array();
        current_snippet["MemoryMappings"] = llvm::json::Array();
      }

      current_snippet["Hex"] = std::string(hex);

      processed_snippets.push_back(
          llvm::json::Value(std::move(current_snippet)));

      if ((file_counter + 1) % blocks_per_json_file == 0) {
        size_t json_file_number = file_counter / blocks_per_json_file;
        bool write_successfully = WriteJsonFile(
            std::move(processed_snippets), json_file_number, json_output_dir);
        if (!write_successfully) return 4;
        processed_snippets.clear();
      }
    }

    if (file_counter % report_progress_every == 0)
      std::cerr << "Finished annotating block #" << file_counter << ".\n";

    file_counter++;
  }

  if (!json_output_dir.empty()) {
    size_t json_file_number = file_counter / blocks_per_json_file;
    bool write_successfully = WriteJsonFile(std::move(processed_snippets),
                                            json_file_number, json_output_dir);
    if (!write_successfully) return 4;
  }

  return 0;
}
