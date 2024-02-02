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

#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "gematria/datasets/bhive_importer.h"
#include "gematria/datasets/find_accessed_addrs.h"
#include "gematria/datasets/find_accessed_addrs_exegesis.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/utils/string.h"
#include "llvm/tools/llvm-exegesis/lib/TargetSelect.h"

constexpr uint64_t kInitialRegVal = 0x10000;
constexpr uint64_t kInitialMemVal = 0x7FFFFFFF;
constexpr std::string_view kRegDefPrefix = "# LLVM-EXEGESIS-DEFREG ";
constexpr std::string_view kMemDefPrefix = "# LLVM-EXEGESIS-MEM-DEF ";
constexpr std::string_view kMemMapPrefix = "# LLVM-EXEGESIS-MEM-MAP ";
constexpr std::string_view kMemNamePrefix = "MEM";

enum class annotator_type { exegesis, fast };

bool AbslParseFlag(absl::string_view text, annotator_type* type,
                   std::string* error) {
  if (text == "exegesis") {
    *type = annotator_type::exegesis;
    return true;
  } else if (text == "fast") {
    *type = annotator_type::fast;
    return true;
  }

  *error = "unknown annotator type";
  return false;
}

std::string AbslUnparseFlag(annotator_type type) {
  switch (type) {
    case annotator_type::exegesis:
      return "exegesis";
    case annotator_type::fast:
      return "fast";
  }
}

ABSL_FLAG(std::string, bhive_csv, "", "Filename of the input BHive CSV file");
ABSL_FLAG(
    std::string, output_dir, "",
    "Directory containing output files that can be executed by llvm-exegesis");
ABSL_FLAG(annotator_type, annotator_implementation, annotator_type::fast,
          "The annotator implementation to use.");

std::optional<gematria::AccessedAddrs> GetAccessedAddrs(
    absl::Span<const uint8_t> basic_block,
    gematria::ExegesisAnnotator* exegesis_annotator) {
  const annotator_type annotator_implementation =
      absl::GetFlag(FLAGS_annotator_implementation);
  if (annotator_implementation == annotator_type::fast) {
    // This will only get the first segfault address.
    auto addrs = gematria::FindAccessedAddrs(basic_block);

    if (!addrs.ok()) return std::nullopt;

    return *addrs;
  } else if (annotator_implementation == annotator_type::exegesis) {
    auto addrs = exegesis_annotator->findAccessedAddrs(
        llvm::ArrayRef(basic_block.begin(), basic_block.end()));

    if (!addrs) return std::nullopt;

    return *addrs;
  }
  return std::nullopt;
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const std::string bhive_filename = absl::GetFlag(FLAGS_bhive_csv);
  if (bhive_filename.empty()) {
    std::cerr << "Error: --bhive_csv is required\n";
    return 1;
  }

  const std::string output_dir = absl::GetFlag(FLAGS_output_dir);
  if (output_dir.empty()) {
    std::cerr << "Error: --output_dir is required\n";
    return 1;
  }

  const annotator_type annotator_implementation =
      absl::GetFlag(FLAGS_annotator_implementation);

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

  llvm::exegesis::InitializeX86ExegesisTarget();

  auto llvm_state_or_error = llvm::exegesis::LLVMState::Create("", "native");
  if (!llvm_state_or_error) {
    std::cerr << "Failed to create LLVMState\n";
    return 1;
  }

  std::unique_ptr<gematria::ExegesisAnnotator> exegesis_annotator = nullptr;
  if (annotator_implementation == annotator_type::exegesis) {
    auto exegesis_annotator_or_error =
        gematria::ExegesisAnnotator::create(*llvm_state_or_error);
    if (!exegesis_annotator_or_error) {
      std::cerr << "Failed to create exegesis annotator\n";
      return 1;
    }
    exegesis_annotator = std::move(*exegesis_annotator_or_error);
  }

  std::ifstream bhive_csv_file(bhive_filename);
  unsigned int file_counter = 0;
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
    auto addrs = GetAccessedAddrs(*bytes, exegesis_annotator.get());

    if (!addrs) {
      std::cerr << "Failed to find addresses for block '" << hex << "\n";
      std::cerr << "Block disassembly:\n";
      for (const auto& instr : proto->machine_instructions()) {
        std::cerr << "\t" << instr.assembly() << "\n";
      }
      continue;
    }

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
      output_file << kMemDefPrefix << kMemNamePrefix << " " << addrs->block_size
                  << " " << initial_mem_val_str << "\n";
    }
    for (const auto& addr : addrs->accessed_blocks) {
      output_file << kMemMapPrefix << kMemNamePrefix << " " << std::dec << addr
                  << "\n";
    }

    // Append disassembled instructions.
    for (const auto& instr : proto->machine_instructions()) {
      output_file << instr.assembly() << "\n";
    }

    file_counter++;
  }
}
