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
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG

#ifdef DEBUG
#define LOG(X) llvm::errs() << X << "\n"
#else
#define LOG(X)
#endif

// Author: Zhan Shi
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

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
      MMI_(dynamic_cast<const llvm::LLVMTargetMachine*>(&target_machine_)) {
  const llvm::MCRegisterInfo& MRI = *target_machine_.getMCRegisterInfo();
  for (llvm::MCPhysReg I = 1, E = MRI.getNumRegs(); I != E; ++I) {
    // Append register definition line.
    llvm::StringRef reg_name = MRI.getName(I);
    name_to_reg_[reg_name.str()] = I;
    // push itself to its own superreg2subreg_ list
    superreg2subreg_[reg_name.str()].push_back(reg_name.str());
    for (auto SuperReg : MRI.superregs(I)) {
      if (MRI.isSubRegister(SuperReg, I)) {
        llvm::StringRef super_reg_name = MRI.getName(SuperReg);
        superreg2subreg_[super_reg_name.str()].push_back(reg_name.str());
      }
    }
  }
  // prettyPrintName2Reg();
  // prettyPrintSuperReg2SubReg();
}

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
  for (llvm::MachineInstr& MI : *MBB) {
    // if MI is a control instruction(ret,branch,jmp), skip it
    if (MI.isInlineAsm() || MI.isTerminator() || MI.isEHLabel()) {
      continue;
    }

    // Assert MI cannot be a CALL instruction
    if (MI.isCall()) {
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
    *basic_block_proto.add_canonicalized_instructions() =
        ProtoFromInstruction(I);
  }
  return basic_block_proto;
}

absl::StatusOr<BasicBlockWithThroughputProto> BHiveImporter::ParseMIRCsvLine(
    std::string_view source_name, std::string_view line, size_t BB_name_index,
    size_t throughput_column_index, double throughput_scaling /*= 1.0*/,
    uint64_t base_address /*= 0*/) {
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
  const std::string_view BB_unique_name = columns[BB_name_index];
  const std::string_view throughput_str = columns[throughput_column_index];

  BasicBlockWithThroughputProto proto;

  absl::StatusOr<BasicBlockProto> block_proto_or_status =
      BasicBlockProtoFromMBBName(BB_unique_name, base_address);
  if (!block_proto_or_status.ok()) return block_proto_or_status.status();

  llvm::StringRef MBB_name_ref(BB_unique_name.data(), BB_unique_name.size());
  // lookup the MBB in the map, if not, return error
  if (name_to_mbb_.find(MBB_name_ref) == name_to_mbb_.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not find MBB with name ", BB_unique_name));
  }

  llvm::MachineBasicBlock* MBB = name_to_mbb_[MBB_name_ref];
  std::string func_name = MBB->getParent()->getName().str();
  assert(func_to_live_intervals_.find(func_name) !=
             func_to_live_intervals_.end() &&
         "Function not found in map");
  addInterferenceGraph(*block_proto_or_status,
                       func_to_live_intervals_[func_name],
                       func_to_live_intervals_[func_name]
                           .BBRangeList[std::string(BB_unique_name)]);
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

absl::StatusOr<bool> BHiveImporter::LoadMIRModule(std::string_view file_name) {
  // clear previous loaded module
  func_to_live_intervals_.clear();
  name_to_mbb_.clear();
  if (mir_module_) {
    for (llvm::Function& F : mir_module_->functions()) {
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
  for (auto& F : *mir_module_) {
    if (F.isDeclaration()) continue;
    llvm::MachineFunction& MF = MMI_.getOrCreateMachineFunction(F);
    for (auto& MBB : MF) {
      // assert name is unique
      if (name_to_mbb_.find(MBB.getName()) != name_to_mbb_.end()) {
        // clear this key-value pair
        name_to_mbb_.erase(MBB.getName());
      } else {
        name_to_mbb_[MBB.getName()] = &MBB;
      }
    }
  }
  return true;
}

// Debug Utilities that prints information of a single RegLiveIntervals
void printRegLiveIntervals(BHiveImporter::RegLiveIntervals LI) {
  LOG("Information of Single RegLiveIntervals named: " << LI.name);

  for (BHiveImporter::BhiveLiveRange range : LI.rangeList) {
    LOG("  Live range: " << range.first << ", " << range.second);
  }

  LOG("  Anchor: " << LI.anchor);
  LOG("  Weight: " << LI.weight);
}

// Debug Utilities for FunctionLiveIntervalInfoMap, which includes
// Name of a function as well as FunctionLiveIntervalInfo
void printMap(
    std::unordered_map<std::string, BHiveImporter::FunctionLiveIntervalInfo>&
        FunctionLiveIntervalInfoMap) {
  // Indicate there is a test
  std::cerr << "*********Start of my test************"
            << "\n";

  for (auto& functionInfoPair : FunctionLiveIntervalInfoMap) {
    std::cerr << "Function Name: " << functionInfoPair.first << "\n";

    // Print live range of register
    for (auto& pairInfo :
         functionInfoPair.second.virtual_register_live_range_func) {
      LOG("Virtual Register Name: " << pairInfo.first);
      printRegLiveIntervals(pairInfo.second);
    }

    for (auto& pairInfo :
         functionInfoPair.second.physical_register_live_range_func) {
      LOG("Physical Register Name: " << pairInfo.first);
      printRegLiveIntervals(pairInfo.second);
    }

    // And also we test the BBrange as well
    for (auto& pairInfo : functionInfoPair.second.BBRangeList) {
      LOG("BB Name: " << pairInfo.first);
      LOG("  Live range: " << pairInfo.second.first << ", "
                           << pairInfo.second.second);
    }

    LOG("-------End of a Function-------");
  }
}

static bool areIntersected(const BHiveImporter::BhiveLiveRange& range1,
                           const BHiveImporter::BhiveLiveRange& range2) {
  // Check if one range starts after the other ends or vice versa.
  if (range1.second <= range2.first || range2.second <= range1.first) {
    return false;  // No intersection.
  }
  return true;  // Ranges are intersected.
}

absl::StatusOr<bool> BHiveImporter::InteferenceGraphParser(
    std::string_view file_name) {
  // Boilerplate for reading input
  std::ifstream input_file{std::string(file_name)};

  if (!input_file.is_open()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not open file ", file_name));
  }
  // FunctionLiveIntervalInfo Denotes all live ranges and bb ranges for a single
  // function FunctionLiveIntervalInfoList is a llvm smal vector that stores
  // FunctionLiveIntervalInfo We read one line at a time
  std::string line;

  // For each function, we need to first store an empty info into the hashmap
  // and then modify its contents while it is in hasmap
  // To do this, we need to have an allias called "ref" that refers to info
  // That is already stored in the hashmap
  FunctionLiveIntervalInfo* info = nullptr;
  std::string curFuncName = "dummy";
  bool isParsingRegister = false;

  // Read each line
  while (std::getline(input_file, line)) {
    std::istringstream lineStream(line);

    // if we encounter a '%' symbol at the beginning, then we encountered a live
    // interval register
    if (isParsingRegister) {
      if (line.substr(0, 8) == "RegMasks") {
        isParsingRegister = false;
        // LOG("End of parsing register");
        continue;
      }
      std::string currentRegister;
      unsigned int start, end, discard;
      char dummy;

      bool is_virtual = line[0] == '%';

      // Get the register name first
      lineStream >> currentRegister;

      // Understand how many life ranges are there in this line
      uint32_t numberLiveRanges = std::count(line.begin(), line.end(), '[');

      // Now we need to read the starting and ending indices of a live range
      for (uint32_t count = 0; count < numberLiveRanges; count++) {
        lineStream >> dummy >> start >> dummy >> dummy >> end >> dummy >>
            dummy >> discard >> dummy;

        // Print out information for debug
        // std::cerr << "Register: " << currentRegister << ", " << start << ", "
        //           << end << "\n";

        // Since LLVM do not support [] operator we need to find it first
        auto resultRegLiveIntervals =
            (is_virtual)
                ? info->virtual_register_live_range_func.find(currentRegister)
                : info->physical_register_live_range_func.find(currentRegister);

        // If you find the current register in the register_live_range_func,
        // you insert a BhiveLiveRange with {start, end} in the range list of
        // the find return If not, then you insert a new pair: {currentRegister,
        // RegLiveIntervals}
        if (is_virtual) {
          if (resultRegLiveIntervals !=
              info->virtual_register_live_range_func.end()) {
            // If you find the register, then you insert a new range
            resultRegLiveIntervals->second.rangeList.push_back({start, end});
          } else {
            // If you do not find the register, then you insert a new pair
            // {currentRegister, RegLiveIntervals}
            RegLiveIntervals newRegLiveIntervals;
            newRegLiveIntervals.name = currentRegister;
            newRegLiveIntervals.rangeList.push_back({start, end});
            info->virtual_register_live_range_func.insert(
                {currentRegister, newRegLiveIntervals});
          }
        } else {
          if (resultRegLiveIntervals !=
              info->physical_register_live_range_func.end()) {
            // If you find the register, then you insert a new range
            resultRegLiveIntervals->second.rangeList.push_back({start, end});
          } else {
            // If you do not find the register, then you insert a new pair
            // {currentRegister, RegLiveIntervals}
            RegLiveIntervals newRegLiveIntervals;
            newRegLiveIntervals.name = currentRegister;
            newRegLiveIntervals.rangeList.push_back({start, end});
            info->physical_register_live_range_func.insert(
                {currentRegister, newRegLiveIntervals});
          }
        }
      }
    }

    // If we encounter a "BB_" symbol, then we encounter a BB entry
    else if (line.substr(0, 3) == "BB_") {
      std::string currentBB;
      unsigned int start, end;
      char dummy;
      std::string junk;

      // Read name of BB and delete the trailing ':'
      lineStream >> currentBB;
      if (currentBB[currentBB.size() - 1] == ':')
        currentBB.erase(currentBB.size() - 1);

      // read range
      lineStream >> start >> dummy >> end;

      info->BBRangeList[currentBB] = {start, end};
    }

    // In this case, we arrived at the definition of a new function
    // In this case we need to
    else if (line[0] == '_') {
      // We reached the end of a function, add info to the Map
      // If this is the beginning of a new function, just add
      // a dummy value and delete it at the end

      std::string copyName(line);
      // Store new function name and information
      lineStream >> curFuncName;
      // LOG("curr Function name is : " << curFuncName);
      func_to_live_intervals_[curFuncName] = FunctionLiveIntervalInfo();
      info = &func_to_live_intervals_[curFuncName];
      isParsingRegister = true;
    }
  }

  // Now we want to debug and print things inside the FunctionLiveIntervalMap
  // printMap(func_to_live_intervals_);
  return true;
}

static bool checkRegIntersectionsWithBBRange(
    const BHiveImporter::RegLiveIntervals& reg_live_interval1,
    const BHiveImporter::RegLiveIntervals& reg_live_interval2,
    const BHiveImporter::BhiveLiveRange& bb_range) {
  const BHiveImporter::BhiveLiveRange* range1HitsBB = nullptr;
  for (auto& interval : reg_live_interval1.rangeList) {
    if (areIntersected(interval, bb_range)) {
      range1HitsBB = &interval;
    }
  }
  if (!range1HitsBB) {
    return false;
  }
  for (auto& interval : reg_live_interval2.rangeList) {
    if (areIntersected(interval, bb_range)) {
      if (areIntersected(*range1HitsBB, interval)) {
        return true;
      }
    }
  }
  return false;
}

void BHiveImporter::addInterferenceGraph(
    BasicBlockProto& bb_proto,
    BHiveImporter::FunctionLiveIntervalInfo& func_live_infos,
    BHiveImporter::BhiveLiveRange& bb_range) {
  std::set<std::string> live_virtual_registers;
  std::set<std::string> live_physical_registers;

  // helper function to update live_virtual_registers and
  // live_physical_registers
  auto update_live_regs = [&](const CanonicalizedOperandProto& operand) {
    if (operand.operand_case() == CanonicalizedOperandProto::kVirtualRegister) {
      live_virtual_registers.insert(operand.virtual_register().name());
    } else if (operand.operand_case() ==
               CanonicalizedOperandProto::kRegisterName) {
      live_physical_registers.insert(operand.register_name());
    } else if (operand.operand_case() == CanonicalizedOperandProto::kAddress) {
      if (!operand.address().base_register().empty()) {
        if (operand.address().base_register()[0] == '%') {
          live_virtual_registers.insert(operand.address().base_register());
        } else {
          live_physical_registers.insert(operand.address().base_register());
        }
      }
      if (!operand.address().index_register().empty()) {
        if (operand.address().index_register()[0] == '%') {
          live_virtual_registers.insert(operand.address().index_register());
        } else {
          live_physical_registers.insert(operand.address().index_register());
        }
      }
      if (!operand.address().segment().empty()) {
        if (operand.address().segment()[0] == '%') {
          live_virtual_registers.insert(operand.address().segment());
        } else {
          live_physical_registers.insert(operand.address().segment());
        }
      }
    }
  };

  auto add_interference_on_name =
      [&](const std::string& name,
          google::protobuf::RepeatedPtrField<std::string>*
              mutable_intefered_register) {
        for (auto vReg : live_virtual_registers) {
          if (vReg == name) continue;
          assert(func_live_infos.virtual_register_live_range_func.find(vReg) !=
                     func_live_infos.virtual_register_live_range_func.end() &&
                 "Virtual register not found in map");
          // If the live range of the two registers intersect, then add
          // interference to proto
          if (checkRegIntersectionsWithBBRange(
                  func_live_infos.virtual_register_live_range_func[name],
                  func_live_infos.virtual_register_live_range_func[vReg],
                  bb_range)) {
            mutable_intefered_register->Add(std::move(vReg));
          }
        }
        // add interference from physical registers to current operand
        for (auto pReg : live_physical_registers) {
          auto subRegs = superreg2subreg_[pReg];
          // if there's one subReg of Preg that has interference with current
          // operand then add interference to proto
          for (auto subReg : subRegs) {
            if (func_live_infos.physical_register_live_range_func.find(
                    subReg) ==
                func_live_infos.physical_register_live_range_func.end())
              continue;
            // pretty print live range of subRegs
            LOG("Live range of subReg: " << subReg);
            for (auto& range :
                 func_live_infos.physical_register_live_range_func[subReg]
                     .rangeList) {
              LOG("  " << range.first << ", " << range.second);
            }
            if (checkRegIntersectionsWithBBRange(
                    func_live_infos.virtual_register_live_range_func[name],
                    func_live_infos.physical_register_live_range_func[subReg],
                    bb_range)) {
              mutable_intefered_register->Add(std::move(pReg));
              break;
            }
          }
        }
      };

  auto add_interference = [&](CanonicalizedOperandProto& operand) {
    if (operand.operand_case() == CanonicalizedOperandProto::kVirtualRegister) {
      add_interference_on_name(operand.virtual_register().name(),
                               operand.mutable_intefered_register());
    } else if (operand.operand_case() == CanonicalizedOperandProto::kAddress) {
      if (!operand.address().base_register().empty() &&
          operand.address().base_register()[0] == '%') {
        add_interference_on_name(
            operand.address().base_register(),
            operand.mutable_address()
                ->mutable_base_register_intefered_register());
      }
      if (!operand.address().index_register().empty() &&
          operand.address().index_register()[0] == '%') {
        add_interference_on_name(
            operand.address().index_register(),
            operand.mutable_address()
                ->mutable_index_register_intefered_register());
      }
      if (!operand.address().segment().empty() &&
          operand.address().segment()[0] == '%') {
        add_interference_on_name(
            operand.address().segment(),
            operand.mutable_address()->mutable_segment_intefered_register());
      }
    }
  };
  // iterate over all operands in bb_proto, add virtual registers to
  // live_virtual_registers
  for (const auto& instruction : bb_proto.canonicalized_instructions()) {
    for (const auto& operand : instruction.input_operands()) {
      update_live_regs(operand);
    }
    for (const auto& operand : instruction.implicit_input_operands()) {
      update_live_regs(operand);
    }
    for (const auto& operand : instruction.output_operands()) {
      update_live_regs(operand);
    }
    for (const auto& operand : instruction.implicit_output_operands()) {
      update_live_regs(operand);
    }
  }

  // pretty print physical registers
  LOG("Physical Registers: ");
  for (auto& reg : live_physical_registers) {
    LOG("Physical Register: " << reg);
  }

  // Iterate over all operands in bb_proto, add interference registers to each
  // operand
  for (auto& instruction : *bb_proto.mutable_canonicalized_instructions()) {
    // LOG("before: " << instruction.DebugString());
    for (auto& operand : *instruction.mutable_input_operands()) {
      add_interference(operand);
    }
    for (auto& operand : *instruction.mutable_output_operands()) {
      add_interference(operand);
    }
    // LOG("after: " << instruction.DebugString());
  }
}

}  // namespace gematria
