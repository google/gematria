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

#include "gematria/datasets/bhive_importer.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "gematria/basic_block/basic_block_protos.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/llvm/llvm_to_absl.h"
#include "gematria/proto/basic_block.pb.h"
#include "gematria/proto/throughput.pb.h"
#include "gematria/utils/string.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"


#define DEBUG

#ifdef DEBUG
#define LOG(X) \
  llvm::errs() << X << "\n"
#else
#define LOG(X)
#endif

//Author: Zhan Shi
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/CodeGen/LiveInterval.h"

namespace gematria {
namespace {

// The assembly syntax used in the output protos. The assembly code in the proto
// is purely informative, so we simply use the "default" assembly syntax for
// each architecture, since it is guaranteed to always be there.
constexpr int kDefaultSyntax = 0;

}  // namespace

BHiveImporter::BHiveImporter(const Canonicalizer* canonicalizer)
    : canonicalizer_(*ABSL_DIE_IF_NULL(canonicalizer)),
      target_machine_(canonicalizer->target_machine()),
      context_(std::make_unique<llvm::MCContext>(
          target_machine_.getTargetTriple(), target_machine_.getMCAsmInfo(),
          target_machine_.getMCRegisterInfo(),
          target_machine_.getMCSubtargetInfo())),
      disassembler_(target_machine_.getTarget().createMCDisassembler(
          *target_machine_.getMCSubtargetInfo(), *context_)),
      mc_inst_printer_(target_machine_.getTarget().createMCInstPrinter(
          target_machine_.getTargetTriple(), kDefaultSyntax,
          *target_machine_.getMCAsmInfo(), *target_machine_.getMCInstrInfo(),
          *target_machine_.getMCRegisterInfo())), 
      MMI_(dynamic_cast<const llvm::LLVMTargetMachine*>(&target_machine_)) {}

absl::StatusOr<BasicBlockProto> BHiveImporter::BasicBlockProtoFromMachineCode(
    llvm::ArrayRef<uint8_t> machine_code, uint64_t base_address /*= 0*/) {
  BasicBlockProto basic_block_proto;
  llvm::Expected<std::vector<DisassembledInstruction>> instructions =
      DisassembleAllInstructions(*disassembler_,
                                 *target_machine_.getMCInstrInfo(),
                                 *target_machine_.getMCRegisterInfo(),
                                 *target_machine_.getMCSubtargetInfo(),
                                 *mc_inst_printer_, base_address, machine_code);
  if (llvm::Error error = instructions.takeError()) {
    return LlvmErrorToStatus(std::move(error));
  }

  for (DisassembledInstruction& instruction : *instructions) {
    MachineInstructionProto& machine_instruction =
        *basic_block_proto.add_machine_instructions();
    machine_instruction.set_address(instruction.address);
    machine_instruction.set_assembly(instruction.assembly);
    machine_instruction.set_machine_code(instruction.machine_code);
    *basic_block_proto.add_canonicalized_instructions() = ProtoFromInstruction(
        canonicalizer_.InstructionFromMCInst(instruction.mc_inst));
  }
  return basic_block_proto;
}

absl::StatusOr<BasicBlockProto>
BHiveImporter::BasicBlockProtoFromMachineCodeHex(
    std::string_view machine_code_hex, uint64_t base_address /*= 0*/) {
  const auto machine_code_bytes_or_status = ParseHexString(machine_code_hex);
  if (!machine_code_bytes_or_status.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("cannot parse: ", machine_code_hex));
  }

  return BasicBlockProtoFromMachineCode(machine_code_bytes_or_status.value(),
                                        base_address);
}

absl::StatusOr<BasicBlockWithThroughputProto> BHiveImporter::ParseBHiveCsvLine(
    std::string_view source_name, std::string_view line,
    size_t machine_code_hex_column_index, size_t throughput_column_index,
    double throughput_scaling /*= 1.0*/, uint64_t base_address /*= 0*/) {
  const absl::InlinedVector<std::string_view, 2> columns =
      absl::StrSplit(line, ',');
  const int min_required_num_columns =
      std::max(machine_code_hex_column_index, throughput_column_index) + 1;
  if (columns.size() < min_required_num_columns) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected `line` to have at least %d columns, found %d: %s",
        min_required_num_columns, columns.size(), line));
  }
  if (machine_code_hex_column_index == throughput_column_index) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected machine code column and throughput column indices to be "
        "different, but were both %d: %s",
        machine_code_hex_column_index, line));
  }
  const std::string_view machine_code_hex =
      columns[machine_code_hex_column_index];
  const std::string_view throughput_str = columns[throughput_column_index];

  BasicBlockWithThroughputProto proto;
  absl::StatusOr<BasicBlockProto> block_proto_or_status =
      BasicBlockProtoFromMachineCodeHex(machine_code_hex, base_address);
  if (!block_proto_or_status.ok()) return block_proto_or_status.status();
  *proto.mutable_basic_block() = std::move(block_proto_or_status).value();

  double throughput_cycles = 0.0;
  if (!absl::SimpleAtod(throughput_str, &throughput_cycles)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not parse throughput value ", throughput_str));
  }

  ThroughputWithSourceProto& throughput = *proto.add_inverse_throughputs();
  throughput.set_source(source_name);
  throughput.add_inverse_throughput_cycles(throughput_cycles *
                                           throughput_scaling);

  return proto;
}

absl::StatusOr<BasicBlockProto> BHiveImporter::BasicBlockProtoFromMBBName(
    std::string_view MBB_name, uint64_t base_address /*= 0*/) {
  BasicBlockProto basic_block_proto;
  // convert MBB_name to llvm::StringRef
  llvm::StringRef MBB_name_ref(MBB_name.data(), MBB_name.size());

  // lookup the MBB in the map, if not, return error
  if (name_to_mbb_.find(MBB_name_ref) == name_to_mbb_.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not find MBB with name ", MBB_name));
  }

  llvm::MachineBasicBlock* MBB = name_to_mbb_[MBB_name_ref];
  LOG("MBB is " << *MBB);
  for (llvm::MachineInstr& MI : *MBB){
    // if MI is a control instruction(ret,branch,jmp), skip it
    if (MI.isInlineAsm() || MI.isTerminator() || MI.isEHLabel()) {
      continue;
    }

    // Assert MI cannot be a CALL instruction
    if(MI.isCall()){
      LOG("MI is a CALL instruction, abort this BB " << MI);
      return absl::InvalidArgumentError(
        absl::StrCat("Cannot handle CALL instruction "));
    } 
    auto I = canonicalizer_.InstructionFromMachineInstr(MI);
    if (!I.is_valid) {
      LOG("MI is not valid, skipping it " << MI);
      return absl::InvalidArgumentError(
        absl::StrCat("Could not parse MachineInstr "));
    }
    *basic_block_proto.add_canonicalized_instructions() = ProtoFromInstruction(I);
  }
  return basic_block_proto;
}

absl::StatusOr<BasicBlockWithThroughputProto> BHiveImporter::ParseMIRCsvLine(
    std::string_view source_name, std::string_view line,
    size_t BB_name_index, size_t throughput_column_index,
    double throughput_scaling /*= 1.0*/, uint64_t base_address /*= 0*/) {
  const absl::InlinedVector<std::string_view, 2> columns =
      absl::StrSplit(line, ',');
  const int min_required_num_columns =
      std::max(BB_name_index, throughput_column_index) + 1;
  if (columns.size() < min_required_num_columns) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected `line` to have at least %d columns, found %d: %s",
        min_required_num_columns, columns.size(), line));
  }
  if (BB_name_index == throughput_column_index) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected BB name column and throughput column indices to be "
        "different, but were both %d: %s",
        BB_name_index, line));
  }
  const std::string_view BB_unique_name =
      columns[BB_name_index];
  const std::string_view throughput_str = columns[throughput_column_index];

  BasicBlockWithThroughputProto proto;

  absl::StatusOr<BasicBlockProto> block_proto_or_status =
      BasicBlockProtoFromMBBName(BB_unique_name, base_address);
  if (!block_proto_or_status.ok()) return block_proto_or_status.status();
  *proto.mutable_basic_block() = std::move(block_proto_or_status).value();

  double throughput_cycles = 0.0;
  if (!absl::SimpleAtod(throughput_str, &throughput_cycles)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not parse throughput value ", throughput_str));
  }

  ThroughputWithSourceProto& throughput = *proto.add_inverse_throughputs();
  throughput.set_source(source_name);
  throughput.add_inverse_throughput_cycles(throughput_cycles *
                                           throughput_scaling);
  LOG(proto.DebugString());

  return proto;
}

static void prettyPrintFunctionLiveIntervalInfo(const BHiveImporter::FunctionLiveIntervalInfo& info){
  for (auto& [name, rangeList] : info.register_name_to_operands){
    LOG("Register name: " << name);
    for (auto& [start, end] : rangeList){
      LOG("Start: " << *start << " End: " << *end);
    }
  }
}
static void prettyPrintFuncToLiveOntervals(const llvm::DenseMap<llvm::StringRef, BHiveImporter::FunctionLiveIntervalInfo>& func_to_live_intervals_){
  for (auto& [name, info] : func_to_live_intervals_){
    LOG("Function name: " << name);
    prettyPrintFunctionLiveIntervalInfo(info);
  }
}

absl::StatusOr<bool> BHiveImporter::LoadMIRModule(std::string_view file_name){
  // clear previous loaded module
  name_to_mbb_.clear();
  if (mir_module_){
    for (llvm::Function &F : mir_module_->functions()) {
        MMI_.deleteMachineFunctionFor(F);
    }
  }

  // create MIR Parser and read all MBB to the map based on their unique name
  llvm::SMDiagnostic diag;

  mir_parser_ = llvm::createMIRParserFromFile(file_name, diag, llvm_context_);
  if (!mir_parser_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not create MIR parser for file ", file_name));
  }

  // Parse the LLVM IR module (if any)
  mir_module_ = mir_parser_->parseIRModule();
  if (!mir_module_) {
      // Handle error
      return absl::InvalidArgumentError(
        absl::StrCat("Could not parse MIR module for file ", file_name));
  }

  MMI_.initialize();

  // Parse the MachineFunctions and add them to MMI
  if (mir_parser_->parseMachineFunctions(*mir_module_, MMI_)) {
      // Handle error
      return absl::InvalidArgumentError(
        absl::StrCat("Could not parse MachineFunctions for file ", file_name));
  }

  // Now iterate over the MachineFunctions and their MachineBasicBlocks
    for (auto &F : *mir_module_) {
        if (F.isDeclaration()) continue;
        llvm::MachineFunction &MF = MMI_.getOrCreateMachineFunction(F);
        assert(func_to_live_intervals_.find(MF.getName()) == func_to_live_intervals_.end() && "Cannot have duplicated function name");
        func_to_live_intervals_[MF.getName()] = FunctionLiveIntervalInfo();
        for (auto &MBB : MF) {
            // assert name is unique
            if (name_to_mbb_.find(MBB.getName()) != name_to_mbb_.end()) {
              // clear this key-value pair
              name_to_mbb_.erase(MBB.getName());
            } else {
              name_to_mbb_[MBB.getName()] = &MBB;
            }
        }
    }

    prettyPrintFuncToLiveOntervals(func_to_live_intervals_);
    return true;
}

// Debug Utilities that prints information of a single RegLiveIntervals
absl::Status printRegLiveIntervals(BHiveImporter::RegLiveIntervals LI) {

  std::cerr << "Information of Single RegLiveIntervals named: " << LI.name << "\n";

  for (BHiveImporter::BhiveLiveRange range: LI.rangeList) {
    std::cerr << "  Live range: " << range.first << ", " << range.second << "\n";
  }

  std::cerr << "  Anchor: " << LI.anchor << "\n";
  std::cerr << "  Weight: " << LI.weight << "\n"; 
  std::cerr << "\n"; 

  // No meaningful value to return.
  return absl::OkStatus();
}

// Debug Utilities for BB
absl::Status printBBRangeList(llvm::StringRef name, llvm::SmallVector<BHiveImporter::BhiveLiveRange> rangeList) {
  std::cerr << "Information of Single RegLiveIntervals named: " << name.str() << "\n";

  for (BHiveImporter::BhiveLiveRange range: rangeList) {
    std::cerr << "  Live range: " << range.first << ", " << range.second << "\n";
  }
  
  // No meaningful value to return.
  return absl::OkStatus();
}

// Debug Utilities for FunctionLiveIntervalInfoMap, which includes
// Name of a function as well as FunctionLiveIntervalInfo
absl::Status printMap(
  llvm::DenseMap<llvm::StringRef, BHiveImporter::FunctionLiveIntervalInfo> FunctionLiveIntervalInfoMap) {
  
  // Indicate there is a test
  std::cerr << "*********Start of my test************" << "\n";
  

  for (auto &functionInfoPair : FunctionLiveIntervalInfoMap) {
    std::cerr << "Function Name: " << functionInfoPair.first.str() << "\n";

    // Print live range of register
    for (auto &pairInfo : functionInfoPair.second.register_live_range_func) 
      printRegLiveIntervals(pairInfo.second);

    // And also we test the BBrange as well
    for (auto &pairInfo : functionInfoPair.second.BBRangeList) 
      printBBRangeList(pairInfo.first, pairInfo.second);

    std::cerr << "-------End of a Function-------" << "\n";
  }
  
  absl::OkStatus();
}

absl::StatusOr<bool> BHiveImporter::InteferenceGraphParser(std::string_view file_name) {
  // Boilerplate for reading input
  std::ifstream input_file{std::string(file_name)};
  
  if (!input_file.is_open())
  {
      return absl::InvalidArgumentError(
      absl::StrCat("Could not open file ", file_name));
  }
  // FunctionLiveIntervalInfo Denotes all live ranges and bb ranges for a single function
  // FunctionLiveIntervalInfoList is a llvm smal vector that stores FunctionLiveIntervalInfo
  // We read one line at a time
  llvm::DenseMap<llvm::StringRef, FunctionLiveIntervalInfo> FunctionLiveIntervalInfoMap; 
  std::string line;

  // For each function, we need to first store an empty info into the hashmap
  // and then modify its contents while it is in hasmap
  // To do this, we need to have an allias called "ref" that refers to info
  // That is already stored in the hashmap
  FunctionLiveIntervalInfo info;
  std::string curFuncName = "dummy";

  // Read each line
  while (std::getline(input_file, line)) {
    std::istringstream lineStream(line);

    // if we encounter a '%' symbol at the beginning, then we encountered a live interval register
    if (line[0] == '%') {
      std::string currentRegister;
      unsigned int start, end;
      char dummy; std::string junk;

      // Get the register name first
      lineStream >> currentRegister;

      // Understand how many life ranges are there in this line
      uint32_t numberLiveRanges = std::count(line.begin(), line.end(), '[');

      // Now we need to read the starting and ending indices of a live range
      for (uint32_t count = 0; count < numberLiveRanges; count++) {
        lineStream >> dummy >> start >> dummy >> dummy >> end >> junk;

        // Print out information for debug
        std::cerr << "Register: " << currentRegister << ", " << start << ", " << end << "\n"; 

        // Since LLVM do not support [] operator we need to find it first
        auto resultRegLiveIntervals = info.register_live_range_func.find(currentRegister);

        // If you find the current register in the register_live_range_func, 
        // you insert a BhiveLiveRange with {start, end} in the range list of the find return
        // If not, then you insert a new pair: {currentRegister, RegLiveIntervals}
        if (resultRegLiveIntervals != info.register_live_range_func.end()) 
          (*resultRegLiveIntervals).second.rangeList.push_back(BhiveLiveRange {start, end});
        else {
          info.register_live_range_func.insert(
            std::pair<llvm::StringRef, RegLiveIntervals> 
              {currentRegister, 
                {currentRegister, 
                llvm::SmallVector<BhiveLiveRange> {{start, end}}, 
                "", 
                "", 
                }
              }
          );
        }
      }
    }
    
    // If we encounter a "BB_" symbol, then we encounter a BB entry
    else if (line.substr(0, 3) == "BB_") {
      std::string currentBB;
      unsigned int start, end;
      char dummy; std::string junk;

      // Read name of BB and delete the trailing ':'
      lineStream >> currentBB;
      if (currentBB[currentBB.size() - 1] == ':') currentBB.erase(currentBB.size() - 1);

      // read range
      lineStream >> start >> dummy >> end;

      // Then insert the value
      // Notice we need to make a copy of BB to insert otherwise we overwrite other values
      // Fix: The current BB name is not correct, got overwritten in each iteration
      std::string stripped(currentBB); 
      info.BBRangeList.insert( 
        std::pair<llvm::StringRef, llvm::SmallVector<BhiveLiveRange>>{
          llvm::StringRef(stripped.c_str(), stripped.length()), 
          llvm::SmallVector<BhiveLiveRange>{{start, end}}
        }
      );
    }

    // In this case, we arrived at the definition of a new function
    // In this case we need to 
    else if (line[0] == '_') {
      // We reached the end of a function, add info to the Map
      // If this is the beginning of a new function, just add
      // a dummy value and delete it at the end

      std::string copyName(curFuncName);
      FunctionLiveIntervalInfoMap.insert(
        std::pair<llvm::StringRef, FunctionLiveIntervalInfo>{
          llvm::StringRef(copyName.c_str(), copyName.length()), 
          info
        }
      );
      
      // Store new function name and information
      lineStream >> curFuncName;

      // If we saw a new function afterwards, we delete the data entry relating to current 
      // function name
      if (FunctionLiveIntervalInfoMap.find(curFuncName) != FunctionLiveIntervalInfoMap.end()) 
        FunctionLiveIntervalInfoMap.erase(curFuncName);
      info = FunctionLiveIntervalInfo();
    }
  }

  // Finally add the final data entry
  FunctionLiveIntervalInfoMap.insert(
    std::pair<llvm::StringRef, FunctionLiveIntervalInfo>{
      llvm::StringRef(curFuncName.c_str(), curFuncName.length()), 
      info
    }
  );

  // Delete dummy node
  FunctionLiveIntervalInfoMap.erase("dummy");

  // Now we want to debug and print things inside the FunctionLiveIntervalMap
  printMap(FunctionLiveIntervalInfoMap);

  // // This stores the information of the whole function
  // std::vector<FunctionInfo> FunctionInfoList; 

  // // At this time, we already processed all information in the file
  // // Now we want to construct the interference graph
  // // We first create an object that represents inference graph in a basic block
  // struct InferenceBB {
  //   std::map<std::string, std::vector<std::string>> adjacencyList;
  // };

  // // This is a vector that stores information of a BB in each function of the function list
  // std::vector<std::vector<InferenceBB>> AllFunction; 

  // // We still need to find what is the name of each basic block
  // for (FunctionInfo functionInfo : FunctionInfoList) {
    
  //   std::vector<InferenceBB> functionAllBB;

  //   // Consider a basic block at a time
  //   for (std::pair<std::string, std::string> BBInformation : functionInfo.BBRangeList ) {
  //     // We create an object that stores adjacency list of a the inference graph of a single BB
  //     InferenceBB adjacencySingleBB;

  //     // Now for each pair of register 
  //     // First decide whether they are in this basic block or not
  //     // and then decide whether they intersect ()
  //     for (RegLiveInterval Reg1 : functionInfo.register_live_range_func) {
  //       for (RegLiveInterval Reg2 : functionInfo.register_live_range_func) {
  //         if (intersect(Reg1, Reg2, BBInformation)) {
  //           adjacencySingleBB.adjacencyList[Reg1.name].push_back(Reg2.name); 
  //           adjacencySingleBB.adjacencyList[Reg2.name].push_back(Reg1.name); 
  //         }
  //       }
  //     }

  //     // Now we add the adjacency of a single BB into the functionAllBB
  //     functionAllBB.push_back(adjacencySingleBB);
  //   }


  //   // Add the inference graph of all BB in a function to the whole list
  //   AllFunction.push_back(functionAllBB);
  return true;
}

}  // namespace gematria
