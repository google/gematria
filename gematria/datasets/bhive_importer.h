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
#include "llvm/CodeGen/MachineModuleInfo.h"


// aUTHOR: Zhan Shi
#include "llvm/Pass.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/CodeGen/MachineDominators.h"

#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"


#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/ADT/DenseMap.h"

#define DEBUG

#ifdef DEBUG
#define LOG(X) \
  llvm::errs() << X << "\n"
#else
#define LOG(X)
#endif

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
  
  absl::StatusOr<BasicBlockProto> BasicBlockProtoFromMBBName(
      std::string_view MBB_name, uint64_t base_address = 0);

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
  
  // Author: Zhan Shi
  // Build the interference graph for each basic block in name_to_mbb_
  // store into name_to_graph_
  void Block_to_Interference() {

    // Use a dense map to store name to the name to graph
    llvm::DenseMap<llvm::StringRef, llvm::MachineBasicBlock*> name_to_graph_;
    
    // Use a pass manager to run the liveness analysis
    // run pass on whole module
    llvm::legacy::PassManager passManager;
    auto lifeIntervalPass = new llvm::LiveIntervals();
    passManager.add(new llvm::MachineDominatorTree());
    passManager.add(lifeIntervalPass);
    passManager.run(*mir_module_);

    // For each function, we want to do live range analysis of this function
    for (auto &F : *mir_module_) {
      // Use a dense map to store live raneges of each register
      llvm::DenseMap<unsigned, llvm::LiveInterval> reg_to_range;

      // We need to get the list of all registers in this function
      llvm::SmallVector<unsigned, 4> reg_list; 
      llvm::MachineFunction &MF = MMI_.getOrCreateMachineFunction(F);
        for (auto &MBB : MF) {
          for (auto& MI : MBB){
            for (unsigned int i = 0; i < MI.getNumOperands(); i++){
              const llvm::MachineOperand& operand = MI.getOperand(i);
              if (operand.isReg()){
                LOG(lifeIntervalPass->getInterval(operand.getReg()));
              }
            }
          }
        }


      // // Find all virtual register used in the function
      // for (unsigned reg : F.all_register) {
      //   // Find live ranges
      //   llvm::LiveInterval &li = LIA.getInterval(reg);

      //   // Add it to the reg_to_range
      //   reg_to_range.insert(std::pair<unsigned, &llvm::LiveInterval>(reg, li));
      // }
      
      // // Now we generate the adjacency list for this function
      // // and then preallocate the space for these inner vectors
      // llvm::DenseMap<unsigned, llvm::SmallVector<unsigned, 4>> adjacency_list;
      // for (unsigned reg : F.all_register) {
      //   llvm::SmallVector<unsigned, 4> temp_small_vec;
      //   adjacency_list.insert(std::pair<unsigned, llvm::SmallVector<unsigned, 4>>(reg, temp_small_vec));
      // }
        

      // // Then we generate all $n choose 2$ case by doing as follows
      // for (std::pair<unsigned, &llvm::LiveInterval> pair_reg_1 : reg_to_range) {
      //   for (std::pair<unsigned, &llvm::LiveInterval> pair_reg_2 : reg_to_range) {

      //     // If we found the register to be the same we do nothing
      //     // Otherwise we do the following
      //     if (pair_reg_1.first != pair_reg_2.first) {
      //       // Find two live ranges
      //       llvm::LiveInterval range_1 = pair_reg_1.second;
      //       llvm::LiveInterval range_2 = pair_reg_2.second;

      //       if (range_1 "intersect" range_2) {
      //         // Retrive the smallvector in the dictionary
      //         llvm::SmallVector<unsigned, 4> &adjacency_1 = adjacency_list.find(pair_reg_1.first) -> second;
      //         llvm::SmallVector<unsigned, 4> &adjacency_2 = adjacency_list.find(pair_reg_2.first) -> second;

      //         // add the other register to the adjacency list of the current node. 
      //         adjacency_1.push_back(pair_reg_2.first); adjacency_2.push_back(pair_reg_1.first); 
      //       }
      //     }
      //   }
      // }
    }
  }
  

 private:
  const Canonicalizer& canonicalizer_;
  const llvm::TargetMachine& target_machine_;
  std::unique_ptr<llvm::MCContext> context_;
  std::unique_ptr<llvm::MCDisassembler> disassembler_;
  std::unique_ptr<llvm::MCInstPrinter> mc_inst_printer_;
  llvm::DenseMap<llvm::StringRef, llvm::MachineBasicBlock*> name_to_mbb_;
  llvm::LLVMContext llvm_context_;
  std::unique_ptr<llvm::Module> mir_module_;
  llvm::MachineModuleInfo MMI_;
  std::unique_ptr<llvm::MIRParser> mir_parser_;

  // Author: Zhan Shi
  // Add one data strcture to the bhiveimporter storing interference graph
  llvm::DenseMap<llvm::StringRef, llvm::MachineBasicBlock*> name_to_graph_;
};


}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_DATASETS_BHIVE_IMPORTER_H_
