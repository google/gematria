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

// Contains a parser for the basic block data set from the BHive repository:
// https://github.com/ithemal/bhive/tree/master/benchmark/throughput.

#ifndef THIRD_PARTY_GEMATRIA_GEMATRIA_DATASETS_BHIVE_IMPORTER_H_
#define THIRD_PARTY_GEMATRIA_GEMATRIA_DATASETS_BHIVE_IMPORTER_H_

#include <cstdint>
#include <memory>
#include <string_view>

#include "absl/status/statusor.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/proto/basic_block.pb.h"
#include "gematria/proto/throughput.pb.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/ADT/DenseMap.h"

namespace gematria {

// Parser for BHive CSV files.
class BHiveImporter {
 public:
  // Creates a new BHive importer from a given canonicalizer. The canonicalizer
  // must be for the architecture/microarchitecture of the data set.
  // Does not take ownership of the canonicalizer.
  explicit BHiveImporter(const Canonicalizer* canonicalizer);

  // Creates a basic block from the given block of machine code. `machine_code`
  // must contain machine code of the instructions to include in the basic
  // block. Expects that the `machine_code.begin()` is the first byte of the
  // first instruction, and `machine_code.rbegin()` is the last byte of the last
  // instruction. Uses `base_address` as the address of the first instruction;
  // the addresses of following instructions are derived from `base_address` and
  // the sizes of the instructions that preceded it.
  // Returns an error when parts of `machine_code` do not disassemble using the
  // provided canonicalizer.

  absl::StatusOr<BasicBlockProto> BasicBlockProtoFromMachineCode(
      llvm::ArrayRef<uint8_t> machine_code, uint64_t base_address = 0);
  // A version of BasicBlockProtoFromMachineCode() where the machine code bytes
  // are provided in the form of a sequence of hex digits, two digits per byte,
  // with no separators between them. For example, the string "AABB11"
  // corresponds to a three-byte sequence {0xAA, 0xBB, 0x11}.
  absl::StatusOr<BasicBlockProto> BasicBlockProtoFromMachineCodeHex(
      std::string_view machine_code_hex, uint64_t base_address = 0);

  // Parses a basic block with throughput from one BHive CSV line. Expects that
  // the line has the format "{machine_code},{throughput}" where {machine_code}
  // is the machine code of the basic block in the hex format accepted by
  // ParseBasicBlockFromMachineCodeHex(), and {throughput} is the inverse
  // throughput of the basic block in text format.
  // Optionally applies `throughput_scaling` to the throughput value, and uses
  // `base_address` as the address of the first instruction in the basic block.
  absl::StatusOr<BasicBlockWithThroughputProto> ParseBHiveCsvLine(
      std::string_view source_name, std::string_view line,
      size_t machine_code_hex_column_index, size_t throughput_column_index,
      double throughput_scaling = 1.0, uint64_t base_address = 0);

  // Parse a file containing machine basic blocks, each has a unique name
  absl::StatusOr<bool> LoadMIRModule(
    std::string_view file_name
  );

  // Parses a MIR basic block with throughput from one BHive CSV line. Expects that
  // the line has the format "{BB_name},{throughput}" where {machine_code}
  // is the machine code of the basic block in the hex format accepted by
  // ParseBasicBlockFromMachineCodeHex(), and {throughput} is the inverse
  // throughput of the basic block in text format.
  // Optionally applies `throughput_scaling` to the throughput value, and uses
  // `base_address` as the address of the first instruction in the basic block. 
  // NOTE: YOU MUST RUN LoadMIRModule before calling this function
  absl::StatusOr<BasicBlockWithThroughputProto> ParseMIRCsvLine(
      std::string_view source_name, std::string_view line,
      size_t BB_name_index, size_t throughput_column_index,
      double throughput_scaling = 1.0, uint64_t base_address = 0);

 private:
  const Canonicalizer& canonicalizer_;
  const llvm::TargetMachine& target_machine_;
  std::unique_ptr<llvm::MCContext> context_;
  std::unique_ptr<llvm::MCDisassembler> disassembler_;
  std::unique_ptr<llvm::MCInstPrinter> mc_inst_printer_;
  llvm::DenseMap<llvm::StringRef, llvm::MachineBasicBlock*> name_to_mbb_;
};

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_DATASETS_BHIVE_IMPORTER_H_
