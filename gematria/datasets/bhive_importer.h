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

#include <sstream>
#include <string>
#include <fstream>      // std::ifstream

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
  // A temporary struct for storing information of live range of a register
    struct RegLiveInterval {
      std::string name; //name of the register
      std::vector< std::pair< std::string, std::string > > rangeList;
      std::string anchor;
      std::string weight;
    };

    // A struct that store all intervals in a function as well as ranges of BB
    struct FunctionInfo {
      std::vector<RegLiveInterval> register_live_range_func;
      std::vector< 
          std::pair< std::string, std::string > > BBRangeList; 
    };

    // Utility for deciding whether two ranges intersect
    bool intersect(RegLiveInterval Reg1, RegLiveInterval Reg2, 
        std::pair<std::string, std::string> BBInformation) {
      
      // First we need to decide the intersection of them
      // WLOG Reg1 starts at an earlier index than Reg2, or otherwise we just swap them
      if (Reg1.rangeList[0].first > Reg2.rangeList[0].first) {
        std::swap(Reg1, Reg2);
      }

      std::pair<std::string, std::string> intersection;
      // Now we need to make sure whether Reg1's end value is later than Reg2
      if (Reg1.rangeList[0].second >= Reg2.rangeList[0].first) {
        // If yes, then they must intersect where intersection begins at Reg2.rangeList.first
        intersection.first = Reg2.rangeList[0].first;

        // We decide the endpoint of the part of intersection
        if (Reg1.rangeList[0].second > Reg2.rangeList[0].second) intersection.second = Reg2.rangeList[0].second; 
        else intersection.second = Reg1.rangeList[0].second;
      }
      // Otherwise they just do not intersect
      else return false;


      // Now given a intersection, we still need to make sure whether the range intersects with the current BB
      bool lowerInBB = (intersection.first <= BBInformation.second) || (intersection.first >= BBInformation.first);
      bool upperInBB = (intersection.second <= BBInformation.second) || (intersection.second >= BBInformation.first);
      return lowerInBB || upperInBB;
    }

  // Now we are able to obtain the live range for each register
  // We want to for each pair of regsiter, find out if their live range overlap
  // Edge case 1: one live range may have multiple live ranges,
  // Non inteference only happens when two register do not overlap on every live range we find
  // Edge case 2: One live register may use part of the bit and the other one use another part
  // Also in constructing the live range we need to take in machine instruction/ fucntion
  void InteferenceGraphParser() {

    //We first reads in to the file
    std::ifstream input_file;
    input_file.open("gematria/datasets/liveintervals_example.text");

    if (!input_file.is_open())
    {
        //fopen returns 0, the NULL pointer, on failure
        perror("Canot open input file\n");
        exit(-1);
    }

    // This stores the information of the whole function
    std::vector<FunctionInfo> FunctionInfoList; 

    // Pass the file stream as a input string stream, simplies the data structure
    // std::istringstream iss(input_file);
    std::string line;

    // This is a temporary FunctionInfo object that captures the information 
    // When we parse a file, we dump every info in a function to this temp
    // If we reach a "**********" then we convey the temporary file to the 
    FunctionInfo temp; 
    bool FirstTime = false; 

    // Now we parse the line
    while (std::getline(input_file, line)) {
      // Create a string stream so that we could process each item in the line
      std::istringstream tempInteval(line);
      // Used as a garbage for something we do not need
      std::string trash;

      // If we reach the line segment, and we it is not the first time 
      if (line.find("**********") != std::string::npos) {
        if (!FirstTime) {
          // Push the temporary variable into the push back
          FunctionInfoList.push_back(temp); 

          // Then we clear up all information in the temp variable for recording 
          // Information of a new function
          FunctionInfo empty; 
          temp = empty;       
        }
          continue;  // Skip the lines and section dividers
        }
      
      // Now if we encounter percent symbol at the beginning, we construct the RegLiveInterval
      // object 
      else if (line[0] == '%') {
        // This means we have an input data that corresponds to a input range

        // We might need register number at the beginning
        // But we throw it away currently
        tempInteval >> trash;

        // Now we put the starting and ending information into the temp
        std::string startInteval; tempInteval >> startInteval; 
        std::string endIntevral;  tempInteval >> endIntevral; 

        std::string anchorIntevral; tempInteval >> anchorIntevral; 
        std::string weight; tempInteval >> weight;

        std::pair<std::string, std::string> oneInterval(startInteval, endIntevral);
        std::vector<std::pair<std::string, std::string>> IntevalListOneReg; 
        IntevalListOneReg.push_back(oneInterval); 

        RegLiveInterval singleInteval = {"name", IntevalListOneReg, anchorIntevral, weight};
        temp.register_live_range_func.push_back(singleInteval);
      }

      else if (line[0] == 'B')
        // In this situation, we encountered information of a basic block
        // Record this information

        // We might need basic block number at the beginning
        // But we throw it away currently
        tempInteval >> trash; 

        // Now we put the starting and ending information into the temp
        std::string startInteval; 
        tempInteval >> startInteval; 
        
        std::string endIntevral; 
        tempInteval >> endIntevral; 

        temp.BBRangeList.push_back(std::pair<std::string, std::string>(startInteval, endIntevral));
    }

    //At the final information to the list
    FunctionInfoList.push_back(temp);

    // At this time, we already processed all information in the file
    // Now we want to construct the interference graph
    // We first create an object that represents inference graph in a basic block
    struct InferenceBB {
      std::map<std::string, std::vector<std::string>> adjacencyList;
    };

    // This is a vector that stores information of a BB in each function of the function list
    std::vector<std::vector<InferenceBB>> AllFunction; 

    // We still need to find what is the name of each basic block
    for (FunctionInfo functionInfo : FunctionInfoList) {
      
      std::vector<InferenceBB> functionAllBB;

      // Consider a basic block at a time
      for (std::pair<std::string, std::string> BBInformation : functionInfo.BBRangeList ) {
        // We create an object that stores adjacency list of a the inference graph of a single BB
        InferenceBB adjacencySingleBB;

        // Now for each pair of register 
        // First decide whether they are in this basic block or not
        // and then decide whether they intersect ()
        for (RegLiveInterval Reg1 : functionInfo.register_live_range_func) {
          for (RegLiveInterval Reg2 : functionInfo.register_live_range_func) {
            if (intersect(Reg1, Reg2, BBInformation)) {
              adjacencySingleBB.adjacencyList[Reg1.name].push_back(Reg2.name); 
              adjacencySingleBB.adjacencyList[Reg2.name].push_back(Reg1.name); 
            }
          }
        }

        // Now we add the adjacency of a single BB into the functionAllBB
        functionAllBB.push_back(adjacencySingleBB);
      }


      // Add the inference graph of all BB in a function to the whole list
      AllFunction.push_back(functionAllBB);
    }
  }

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


      // Now we are able to obtain the live range for each register
      // We want to for each pair of regsiter, find out if their live range overlap
      // Edge case 1: one live range may have multiple live ranges,
      // Non inteference only happens when two register do not overlap on every live range we find
      // Edge case 2: One live register may use part of the bit and the other one use another part
      // Also in constructing the live range we need to take in machine instruction/ fucntion



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
