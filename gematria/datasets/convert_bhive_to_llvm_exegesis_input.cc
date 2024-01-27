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
#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/utils/string.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

constexpr uint64_t kInitialRegVal = 0x10000;
constexpr uint64_t kInitialMemVal = 0x7FFFFFFF;
constexpr std::string_view kRegDefPrefix = "# LLVM-EXEGESIS-DEFREG ";
constexpr std::string_view kMemDefPrefix = "# LLVM-EXEGESIS-MEM-DEF ";
constexpr std::string_view kMemMapPrefix = "# LLVM-EXEGESIS-MEM-MAP ";
constexpr std::string_view kMemNamePrefix = "MEM";

namespace {
unsigned int file_counter = 0;
}

ABSL_FLAG(std::string, bhive_csv, "", "Filename of the input BHive CSV file");
ABSL_FLAG(
    std::string, output_dir, "",
    "Directory containing output files that can be executed by llvm-exegesis");
ABSL_FLAG(std::string, json_output_dir, "",
          "Directory containing JSON output files");
ABSL_FLAG(
    unsigned, json_split_count, std::numeric_limits<unsigned>::max(),
    "The number of annotated basic blocks to include in a single JSON file");
ABSL_FLAG(std::vector<std::string>, output_types,
          std::vector<std::string>({"asm"}),
          "A comma separated list of output types to generate");

bool write_json_file(llvm::json::Array to_write, size_t json_file_number,
                     std::string json_output_dir) {
  llvm::Twine json_output_file_path = llvm::Twine(json_output_dir)
                                          .concat("/")
                                          .concat(llvm::Twine(json_file_number))
                                          .concat(".json");
  std::error_code file_ec;
  llvm::raw_fd_ostream json_output_file(json_output_file_path.str(), file_ec);

  if (file_ec) {
    std::cerr << "Failed to open output file: " << json_output_file_path.str()
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
  const std::string output_dir = absl::GetFlag(FLAGS_output_dir);
  const std::vector<std::string> output_types =
      absl::GetFlag(FLAGS_output_types);
  bool json_output_enabled = false;
  bool asm_output_enabled = false;
  for (const std::string& output_type : output_types) {
    if (output_type == "json")
      json_output_enabled = true;
    else if (output_type == "asm")
      asm_output_enabled = true;
  }

  if (json_output_enabled && json_output_dir.empty()) {
    std::cerr << "Error: --json_output_dir is required when the json output "
                 "type is requested\n";
    return 1;
  } else if (asm_output_enabled && output_dir.empty()) {
    std::cerr << "Error: --output_dir is required when the asm output type is "
                 "requested\n";
    return 1;
  }

  std::string initial_reg_val_str =
      gematria::ConvertHexToString(kInitialRegVal);
  std::string initial_mem_val_str =
      gematria::ConvertHexToString(kInitialMemVal);
  std::string register_defs_lines;
  const std::unique_ptr<gematria::LlvmArchitectureSupport> llvm_support =
      gematria::LlvmArchitectureSupport::X86_64();
  const llvm::MCRegisterInfo& reg_info = llvm_support->mc_register_info();

  // Iterate through all general purpose registers and vector registers
  // and add them to the register definitions.
  // TODO(9Temptest): Change GR64_NOREXRegClassID to GR64_NOREX2RegClassID when
  // the LLVM version is bumped to avoid including the new APX GPRs (r16-r31)
  // that have recently been added to LLVM.
  for (unsigned i = 0;
       i < reg_info.getRegClass(llvm::X86::GR64_NOREXRegClassID).getNumRegs();
       ++i) {
    llvm::StringRef reg_name = reg_info.getName(
        reg_info.getRegClass(llvm::X86::GR64_NOREXRegClassID).getRegister(i));
    register_defs_lines += llvm::Twine(kRegDefPrefix)
                               .concat(reg_name)
                               .concat(" ")
                               .concat(initial_reg_val_str)
                               .concat("\n")
                               .str();
  }
  for (unsigned i = 0;
       i < reg_info.getRegClass(llvm::X86::VR128RegClassID).getNumRegs(); ++i) {
    llvm::StringRef reg_name = reg_info.getName(
        reg_info.getRegClass(llvm::X86::VR128RegClassID).getRegister(i));
    register_defs_lines += llvm::Twine(kRegDefPrefix)
                               .concat(reg_name)
                               .concat(" ")
                               .concat(initial_reg_val_str)
                               .concat("\n")
                               .str();
  }

  gematria::X86Canonicalizer canonicalizer(&llvm_support->target_machine());
  gematria::BHiveImporter bhive_importer(&canonicalizer);

  std::ifstream bhive_csv_file(bhive_filename);
  llvm::json::Array processed_snippets;
  const int json_split_count = absl::GetFlag(FLAGS_json_split_count);
  for (std::string line; std::getline(bhive_csv_file, line);) {
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

    // This will only get the first segfault address.
    auto addrs = gematria::FindAccessedAddrs(*bytes);

    if (!addrs.ok()) {
      std::cerr << "Failed to find addresses for block '" << hex
                << "': " << addrs.status() << "\n";
      std::cerr << "Block disassembly:\n";
      for (const auto& instr : proto->machine_instructions()) {
        std::cerr << "\t" << instr.assembly() << "\n";
      }
      continue;
    }

    if (asm_output_enabled) {
      // Create output file path.
      llvm::Twine output_file_path = llvm::Twine(output_dir)
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

    if (json_output_enabled) {
      llvm::json::Object current_snippet;

      if (addrs->accessed_blocks.size() > 0) {
        llvm::json::Array memory_definitions;
        llvm::json::Object current_memory_definition;
        current_memory_definition["Name"] = llvm::json::Value(kMemNamePrefix);
        current_memory_definition["Size"] =
            llvm::json::Value(addrs->block_size);
        current_memory_definition["Value"] = llvm::json::Value(kInitialMemVal);
        memory_definitions.push_back(std::move(current_memory_definition));
        current_snippet["MemoryDefinitions"] =
            llvm::json::Value(std::move(memory_definitions));

        llvm::json::Array memory_mappings;
        for (const uintptr_t addr : addrs->accessed_blocks) {
          llvm::json::Object current_memory_mapping;
          current_memory_mapping["Value"] = llvm::json::Value(kMemNamePrefix);
          current_memory_mapping["Address"] = llvm::json::Value(addr);
          memory_mappings.push_back(std::move(current_memory_mapping));
        }
        current_snippet["MemoryMappings"] =
            llvm::json::Value(std::move(memory_mappings));
      } else {
        current_snippet["MemoryDefinitions"] = llvm::json::Array();
        current_snippet["MemoryMappings"] = llvm::json::Array();
      }

      std::string hex_string = {hex.begin(), hex.end()};
      current_snippet["Hex"] = llvm::json::Value(hex_string);

      processed_snippets.push_back(
          llvm::json::Value(std::move(current_snippet)));

      if (file_counter % json_split_count == 0) {
        size_t json_file_number = file_counter / json_split_count;
        bool write_successfully = write_json_file(
            std::move(processed_snippets), json_file_number, json_output_dir);
        if (!write_successfully) return 4;
        processed_snippets.clear();
      }
    }

    file_counter++;
  }

  if (json_output_enabled) {
    size_t json_file_number = file_counter / json_split_count;
    bool write_successfully = write_json_file(
        std::move(processed_snippets), json_file_number, json_output_dir);
    if (!write_successfully) return 4;
  }
}
