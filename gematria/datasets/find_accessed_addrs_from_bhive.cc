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
#include <string>
#include <string_view>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "gematria/datasets/find_accessed_addrs.h"
#include "gematria/utils/string.h"

ABSL_FLAG(std::string, bhive_csv, "", "Filename of the input BHive CSV file");
ABSL_FLAG(bool, failures_only, false,
          "Only produce output for blocks which FindAccessedAddrs fails on");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  std::string bhive_filename = absl::GetFlag(FLAGS_bhive_csv);
  if (bhive_filename.empty()) {
    std::cerr << "Error: --bhive_csv is required\n";
    return 1;
  }

  std::ifstream bhive_csv_file(bhive_filename);
  int successful_calls = 0;
  int total_calls = 0;
  for (std::string line; std::getline(bhive_csv_file, line);) {
    auto comma_index = line.find(',');
    if (comma_index == std::string::npos) {
      std::cerr << "Invalid CSV file: no comma in line '" << line << "'\n";
      return 2;
    }

    std::string_view hex = std::string_view(line).substr(0, comma_index);
    auto bytes_or = gematria::ParseHexString(hex);
    if (!bytes_or.ok()) {
      std::cerr << bytes_or.status() << "\n";
      return 3;
    }

    const auto& bytes = bytes_or.value();
    auto addrs_or = gematria::FindAccessedAddrs(bytes);
    if (addrs_or.ok()) {
      successful_calls++;

      if (!absl::GetFlag(FLAGS_failures_only)) {
        auto addrs = addrs_or.value();
        std::cout << "Successfully found addresses for block '" << hex << "'"
                  << ". When mapped at 0x" << std::hex << addrs.code_location
                  << ", block accesses addresses in " << std::dec
                  << addrs.accessed_blocks.size() << " chunk(s) of size 0x"
                  << std::hex << addrs.block_size << ":";

        for (const auto& addr : addrs.accessed_blocks) {
          std::cout << " 0x" << addr;

          if (&addr != &addrs.accessed_blocks.back()) std::cout << ",";
        }
        std::cout << "\n";
      }
    } else {
      std::cerr << "Failed to find addresses for block '" << hex
                << "': " << addrs_or.status() << "\n";
    }
    total_calls++;
  }

  std::cout << "Called FindAccessedAddrs successfully on " << std::dec
            << successful_calls << " / " << total_calls << " blocks\n";
  return 0;
}
