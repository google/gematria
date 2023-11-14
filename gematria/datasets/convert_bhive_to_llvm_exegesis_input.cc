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
#include <memory>
#include <sstream>
#include <string>
#include <string_view>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "gematria/datasets/bhive_importer.h"
#include "gematria/datasets/find_accessed_addrs.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/utils/string.h"

#define INITIAL_REG_VALUE 0x10000
#define INITIAL_MEM_VALUE 2147483647
#define LLVM_EXEGESIS_MEM_DEF_PREFIX "# LLVM-EXEGESIS-MEM-DEF "
#define LLVM_EXEGESIS_MEM_MAP_PREFIX "# LLVM-EXEGESIS-MEM-MAP "
#define LLVM_EXEGESIS_MEM_NAME_PREFIX "MEM"

static unsigned int file_counter = 0;

ABSL_FLAG(std::string, bhive_csv, "",
          "Filename of the input file containing code hex");
ABSL_FLAG(
    std::string, output_dir, "",
    "Directory containing output files that can be executed by llvm-exegesis");
ABSL_FLAG(std::string, exegesis_template,
          "gematria/datasets/llvm-exegesis_warpper.S",
          "Template file for llvm-exegesis");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  std::string bhive_filename = absl::GetFlag(FLAGS_bhive_csv);
  if (bhive_filename.empty()) {
    std::cerr << "Error: --bhive_csv is required\n";
    return 1;
  }

  std::string output_dir = absl::GetFlag(FLAGS_output_dir);
  if (output_dir.empty()) {
    std::cerr << "Error: --output_dir is required\n";
    return 1;
  }

  // open template llvm-exegesis file
  std::string template_file_path = absl::GetFlag(FLAGS_exegesis_template);
  std::string template_content;

  // Read the content of the template file
  {
    std::ifstream template_file(template_file_path);
    std::stringstream buffer;
    buffer << template_file.rdbuf();
    template_content = buffer.str();
  }

  const std::unique_ptr<gematria::LlvmArchitectureSupport> llvm_support =
      gematria::LlvmArchitectureSupport::X86_64();
  gematria::X86Canonicalizer canonicalizer(&llvm_support->target_machine());
  gematria::BHiveImporter bhive_importer(&canonicalizer);

  std::ifstream bhive_csv_file(bhive_filename);
  for (std::string line; std::getline(bhive_csv_file, line);) {
    auto comma_index = line.find(',');
    if (comma_index == std::string::npos) {
      std::cerr << "Invalid CSV file: no comma in line '" << line << "'\n";
      return 2;
    }

    std::string_view hex = std::string_view(line).substr(0, comma_index);
    auto bytes_or = gematria::ParseHexString(hex);
    if (!bytes_or.has_value()) {
      std::cerr << "could not parse: " << hex << "\n";
      return 3;
    }

    // for each line, find the accessed addresses & disassemble instructions
    const auto& bytes = bytes_or.value();

    // this will only get the first segfault address
    auto addrs_or = gematria::FindAccessedAddrs(bytes);
    auto proto = bhive_importer.BasicBlockProtoFromMachineCode(bytes);
    if (addrs_or.ok() && proto.ok()) {
      // Create output file path
      std::string output_file_path =
          output_dir + "/" + std::to_string(file_counter) + ".test";

      // Open output file for writing
      std::ofstream output_file(output_file_path);
      if (!output_file.is_open()) {
        std::cerr << "Failed to open output file: " << output_file_path << "\n";
        return 4;
      }

      // Write the template content into the output file
      output_file << template_content;

      // Append memory annotations
      auto addrs = addrs_or.value();

      // Multiple mappings can point to the same definition
      if (addrs.accessed_blocks.size() > 0) {
        output_file << LLVM_EXEGESIS_MEM_DEF_PREFIX
                    << LLVM_EXEGESIS_MEM_NAME_PREFIX << " " << addrs.block_size
                    << " " << INITIAL_MEM_VALUE << "\n";
      }
      for (const auto& addr : addrs.accessed_blocks) {
        output_file << LLVM_EXEGESIS_MEM_MAP_PREFIX
                    << LLVM_EXEGESIS_MEM_NAME_PREFIX << " " << std::dec << addr
                    << "\n";
      }

      // Append disassembled instructions
      for (const auto& instr : proto->machine_instructions()) {
        output_file << instr.assembly() << "\n";
      }

      file_counter++;
    } else if (proto.ok()) {
      std::cerr << "Failed to find addresses for block '" << hex
                << "': " << addrs_or.status() << "\n";
      std::cerr << "Block disassembly:\n";
      for (const auto& instr : proto->machine_instructions()) {
        std::cerr << "\t" << instr.assembly() << "\n";
      }
    } else {
      std::cerr << "Failed to disassemble block '" << hex << "\n";
    }
  }
}