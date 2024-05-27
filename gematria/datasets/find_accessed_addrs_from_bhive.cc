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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "gematria/datasets/bhive_importer.h"
#include "gematria/datasets/find_accessed_addrs.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/utils/string.h"

ABSL_FLAG(std::string, bhive_csv, "", "Filename of the input BHive CSV file");
ABSL_FLAG(bool, failures_only, false,
          "Only produce output for blocks which FindAccessedAddrs fails on");
ABSL_FLAG(bool, quiet, false, "Omit all output except for the final summary");
ABSL_FLAG(std::string, failing_blocks_csv, "",
          "Filename of an output CSV file to which any failing blocks are "
          "written. This can be used as an input for subsequent runs.");
ABSL_FLAG(std::string, exegesis_snippets_dir, "",
          "Directory to write EXEgesis snippets to");

void WriteRegisterDef(std::ofstream& snippets_file, std::string_view name,
                      std::optional<int64_t> value) {
  if (!value) return;
  snippets_file << "# LLVM-EXEGESIS-DEFREG " << name << " "
                << absl::StrFormat("%016x", *value) << "\n";
}

void WriteExegesisSnippet(gematria::BHiveImporter& bhive_importer,
                          std::string_view snippets_dir,
                          const std::vector<uint8_t>& code,
                          const gematria::AccessedAddrs& addrs, int n) {
  auto proto = bhive_importer.BasicBlockProtoFromMachineCode(code);
  CHECK_OK(proto);

  auto filename = absl::StrFormat("%s/%d", snippets_dir, n);
  std::ofstream snippets_file(filename);

  // register values
  WriteRegisterDef(snippets_file, "RAX", addrs.initial_regs.rax);
  WriteRegisterDef(snippets_file, "RBX", addrs.initial_regs.rbx);
  WriteRegisterDef(snippets_file, "RCX", addrs.initial_regs.rcx);
  WriteRegisterDef(snippets_file, "RDX", addrs.initial_regs.rdx);
  WriteRegisterDef(snippets_file, "RSI", addrs.initial_regs.rsi);
  WriteRegisterDef(snippets_file, "RDI", addrs.initial_regs.rdi);
  WriteRegisterDef(snippets_file, "RSP", addrs.initial_regs.rsp);
  WriteRegisterDef(snippets_file, "RBP", addrs.initial_regs.rbp);
  WriteRegisterDef(snippets_file, "R8", addrs.initial_regs.r8);
  WriteRegisterDef(snippets_file, "R9", addrs.initial_regs.r9);
  WriteRegisterDef(snippets_file, "R10", addrs.initial_regs.r10);
  WriteRegisterDef(snippets_file, "R11", addrs.initial_regs.r11);
  WriteRegisterDef(snippets_file, "R12", addrs.initial_regs.r12);
  WriteRegisterDef(snippets_file, "R13", addrs.initial_regs.r13);
  WriteRegisterDef(snippets_file, "R14", addrs.initial_regs.r14);
  WriteRegisterDef(snippets_file, "R15", addrs.initial_regs.r15);

  // Every block has the same size and contents, so we define one and then map
  // it for every accessed block.
  snippets_file << "# LLVM-EXEGESIS-MEM-DEF block " << addrs.block_size << " "
                << absl::StrFormat("%016x", addrs.block_contents) << "\n";
  for (const auto& addr : addrs.accessed_blocks) {
    snippets_file << "# LLVM-EXEGESIS-MEM-MAP block " << addr << "\n";
  }

  for (const auto& instr : proto->machine_instructions()) {
    snippets_file << instr.assembly() << "\n";
  }
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  std::string bhive_filename = absl::GetFlag(FLAGS_bhive_csv);
  if (bhive_filename.empty()) {
    std::cerr << "Error: --bhive_csv is required\n";
    return 1;
  }

  const bool print_failures = !absl::GetFlag(FLAGS_quiet);
  const bool print_successes =
      !absl::GetFlag(FLAGS_failures_only) && !absl::GetFlag(FLAGS_quiet);

  const std::unique_ptr<gematria::LlvmArchitectureSupport> llvm_support =
      gematria::LlvmArchitectureSupport::X86_64();
  gematria::X86Canonicalizer canonicalizer(&llvm_support->target_machine());
  gematria::BHiveImporter bhive_importer(&canonicalizer);

  std::optional<std::ofstream> failing_blocks_csv_file;
  if (!absl::GetFlag(FLAGS_failing_blocks_csv).empty()) {
    failing_blocks_csv_file =
        std::ofstream(absl::GetFlag(FLAGS_failing_blocks_csv));
  }

  std::ifstream bhive_csv_file(bhive_filename);
  int successful_calls = 0;
  int total_calls = 0;
  int n = 0;
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

    const auto& bytes = bytes_or.value();
    auto addrs_or = gematria::FindAccessedAddrs(bytes);
    if (addrs_or.ok()) {
      successful_calls++;

      if (print_successes) {
        auto addrs = addrs_or.value();
        std::cout << "Successfully found addresses for block '" << hex << "'"
                  << ". When mapped at 0x" << std::hex << addrs.code_location
                  << ", block accesses addresses in " << std::dec
                  << addrs.accessed_blocks.size() << " chunk(s) of size 0x"
                  << std::hex << addrs.block_size << std::dec << ":";

        for (const auto& addr : addrs.accessed_blocks) {
          std::cout << " 0x" << addr;

          if (&addr != &addrs.accessed_blocks.back()) std::cout << ",";
        }
        std::cout << "\n";
      }

      if (!absl::GetFlag(FLAGS_exegesis_snippets_dir).empty()) {
        WriteExegesisSnippet(bhive_importer,
                             absl::GetFlag(FLAGS_exegesis_snippets_dir), bytes,
                             addrs_or.value(), n++);
      }
    } else {
      if (failing_blocks_csv_file.has_value()) {
        *failing_blocks_csv_file << line << "\n";
      }
      if (print_failures) {
        std::cerr << "Failed to find addresses for block '" << hex
                  << "': " << addrs_or.status() << "\n";
        auto proto = bhive_importer.BasicBlockProtoFromMachineCode(bytes);
        if (proto.ok()) {
          std::cerr << "Block disassembly:\n";
          for (const auto& instr : proto->machine_instructions()) {
            std::cerr << "\t" << instr.assembly() << "\n";
          }
        }
      }
    }

    total_calls++;
  }

  std::cout << "Called FindAccessedAddrs successfully on " << std::dec
            << successful_calls << " / " << total_calls << " blocks\n";
  return 0;
}
