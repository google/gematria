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
#include "gematria/datasets/basic_block_utils.h"
#include "gematria/datasets/bhive_importer.h"
#include "gematria/datasets/find_accessed_addrs.h"
#include "gematria/datasets/find_accessed_addrs_exegesis.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "gematria/llvm/llvm_to_absl.h"
#include "gematria/utils/string.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/tools/llvm-exegesis/lib/TargetSelect.h"

constexpr std::string_view kRegDefPrefix = "# LLVM-EXEGESIS-DEFREG ";
constexpr std::string_view kMemDefPrefix = "# LLVM-EXEGESIS-MEM-DEF ";
constexpr std::string_view kMemMapPrefix = "# LLVM-EXEGESIS-MEM-MAP ";
constexpr std::string_view kLoopRegisterPrefix =
    "# LLVM-EXEGESIS-LOOP-REGISTER ";
constexpr std::string_view kMemNamePrefix = "MEM";

enum class AnnotatorType { kExegesis, kFast, kNone };

constexpr std::pair<AnnotatorType, std::string_view> kAnnotatorTypeNames[] = {
    {AnnotatorType::kExegesis, "exegesis"},
    {AnnotatorType::kFast, "fast"},
    {AnnotatorType::kNone, "none"}};

bool AbslParseFlag(absl::string_view text, AnnotatorType* type,
                   std::string* error) {
  for (const auto& [annotator_type, type_string] : kAnnotatorTypeNames) {
    if (text == type_string) {
      *type = annotator_type;
      return true;
    }
  }

  *error = "unknown annotator type";
  return false;
}

std::string AbslUnparseFlag(AnnotatorType type) {
  for (const auto& [annotator_type, type_string] : kAnnotatorTypeNames) {
    if (annotator_type == type) return std::string(type_string);
  }

  __builtin_unreachable();
}

ABSL_FLAG(std::string, bhive_csv, "", "Filename of the input BHive CSV file");
ABSL_FLAG(
    std::string, asm_output_dir, "",
    "Directory containing output files that can be executed by llvm-exegesis");
ABSL_FLAG(AnnotatorType, annotator_implementation, AnnotatorType::kFast,
          "The annotator implementation to use.");
ABSL_FLAG(std::string, json_output_dir, "",
          "Directory containing JSON output files");
ABSL_FLAG(
    unsigned, blocks_per_json_file, std::numeric_limits<unsigned>::max(),
    "The number of annotated basic blocks to include in a single JSON file");
ABSL_FLAG(unsigned, max_bb_count, std::numeric_limits<unsigned>::max(),
          "The maximum number of basic blocks to process");
ABSL_FLAG(unsigned, report_progress_every, std::numeric_limits<unsigned>::max(),
          "The number of blocks after which to report progress.");
ABSL_FLAG(bool, skip_no_loop_register, true,
          "Whether or not to skip basic blocks where a loop counter register "
          "cannot be found.");
ABSL_FLAG(unsigned, max_annotation_attempts, 50,
          "The maximum number of times to attempt to annotate a block before "
          "giving up.");

absl::StatusOr<gematria::AccessedAddrs> GetAccessedAddrs(
    absl::Span<const uint8_t> basic_block,
    gematria::ExegesisAnnotator* exegesis_annotator,
    const unsigned max_annotation_attempts,
    gematria::LlvmArchitectureSupport& llvm_support) {
  const AnnotatorType annotator_implementation =
      absl::GetFlag(FLAGS_annotator_implementation);
  switch (annotator_implementation) {
    case AnnotatorType::kFast:
      return gematria::FindAccessedAddrs(basic_block, llvm_support);
    case AnnotatorType::kExegesis:
      return gematria::LlvmExpectedToStatusOr(
          exegesis_annotator->findAccessedAddrs(
              llvm::ArrayRef(basic_block.begin(), basic_block.end()),
              max_annotation_attempts));
    case AnnotatorType::kNone:
      return gematria::AccessedAddrs();
  }
  return absl::InvalidArgumentError("unknown annotator type");
}

bool WriteJsonFile(llvm::json::Array to_write, size_t json_file_number,
                   llvm::StringRef json_output_dir) {
  llvm::SmallString<40> json_output_file_path(json_output_dir);
  llvm::sys::path::append(json_output_file_path,
                          llvm::Twine(json_file_number).concat(".json"));
  std::error_code file_ec;
  llvm::raw_fd_ostream json_output_file(json_output_file_path, file_ec);

  if (file_ec) {
    std::cerr << "Failed to open output file: "
              << static_cast<std::string_view>(json_output_file_path.str())
              << "\n";
    return false;
  }

  json_output_file
      << llvm::formatv("{0:2}", llvm::json::Value(std::move(to_write))).str();
  return true;
}

struct AnnotatedBlock {
  gematria::AccessedAddrs accessed_addrs;
  gematria::BasicBlockProto basic_block_proto;
  std::optional<unsigned> loop_register;
};

absl::StatusOr<AnnotatedBlock> AnnotateBasicBlock(
    std::string_view basic_block_hex, gematria::BHiveImporter& bhive_importer,
    gematria::ExegesisAnnotator* exegesis_annotator,
    gematria::LlvmArchitectureSupport& llvm_support,
    llvm::MCInstPrinter& inst_printer, const unsigned max_annotation_attempts) {
  auto bytes = gematria::ParseHexString(basic_block_hex);
  if (!bytes.has_value())
    return absl::InvalidArgumentError(
        Twine("Could not parse ").concat(basic_block_hex).str());

  llvm::Expected<std::vector<gematria::DisassembledInstruction>> instructions =
      gematria::DisassembleAllInstructions(
          llvm_support.mc_disassembler(), llvm_support.mc_instr_info(),
          llvm_support.mc_register_info(), llvm_support.mc_subtarget_info(),
          inst_printer, 0, *bytes);

  if (!instructions) {
    return absl::InvalidArgumentError(
        Twine("Failed to disassemble block ").concat(basic_block_hex).str());
  }

  auto proto = bhive_importer.BasicBlockProtoFromInstructions(*instructions);

  auto addrs = GetAccessedAddrs(*bytes, exegesis_annotator,
                                max_annotation_attempts, llvm_support);

  if (!addrs.ok()) return addrs.status();

  AnnotatedBlock annotated_block;
  annotated_block.accessed_addrs = std::move(*addrs);
  annotated_block.basic_block_proto = std::move(proto);
  annotated_block.loop_register = gematria::getUnusedGPRegister(
      *instructions, llvm_support.mc_register_info(),
      llvm_support.mc_instr_info());

  return std::move(annotated_block);
}

llvm::json::Value GetJSONForSnippet(const AnnotatedBlock& annotated_block,
                                    std::string_view hex) {
  llvm::json::Object current_snippet;

  llvm::json::Array register_definitions;
  for (const gematria::RegisterAndValue register_and_value :
       annotated_block.accessed_addrs.initial_regs) {
    llvm::json::Object current_register_definition;
    current_register_definition["Register"] = register_and_value.register_index;
    current_register_definition["Value"] = register_and_value.register_value;
    register_definitions.push_back(std::move(current_register_definition));
  }
  current_snippet["RegisterDefinitions"] =
      llvm::json::Value(std::move(register_definitions));

  // Output the loop register.
  if (annotated_block.loop_register)
    current_snippet["LoopRegister"] = *annotated_block.loop_register;
  else
    current_snippet["LoopRegister"] = llvm::MCRegister::NoRegister;

  if (annotated_block.accessed_addrs.accessed_blocks.size() > 0) {
    llvm::json::Array memory_definitions;
    llvm::json::Object current_memory_definition;
    current_memory_definition["Name"] = llvm::json::Value(kMemNamePrefix);
    current_memory_definition["Size"] =
        annotated_block.accessed_addrs.block_size;
    current_memory_definition["Value"] =
        annotated_block.accessed_addrs.block_contents;
    memory_definitions.push_back(std::move(current_memory_definition));
    current_snippet["MemoryDefinitions"] =
        llvm::json::Value(std::move(memory_definitions));

    llvm::json::Array memory_mappings;
    for (const uintptr_t addr :
         annotated_block.accessed_addrs.accessed_blocks) {
      llvm::json::Object current_memory_mapping;
      current_memory_mapping["Value"] = llvm::json::Value(kMemNamePrefix);
      current_memory_mapping["Address"] = addr;
      memory_mappings.push_back(std::move(current_memory_mapping));
    }
    current_snippet["MemoryMappings"] =
        llvm::json::Value(std::move(memory_mappings));
  } else {
    current_snippet["MemoryDefinitions"] = llvm::json::Array();
    current_snippet["MemoryMappings"] = llvm::json::Array();
  }

  current_snippet["Hex"] = std::string(hex);

  return llvm::json::Value(std::move(current_snippet));
}

absl::Status WriteAsmOutput(const AnnotatedBlock& annotated_block,
                            llvm::StringRef asm_output_dir,
                            unsigned int file_counter,
                            const llvm::MCRegisterInfo& reg_info) {
  // Create output file path.
  llvm::Twine output_file_path = llvm::Twine(asm_output_dir)
                                     .concat("/")
                                     .concat(llvm::Twine(file_counter))
                                     .concat(".test");

  // Open output file for writing.
  std::ofstream output_file(output_file_path.str());
  if (!output_file.is_open()) {
    return absl::InvalidArgumentError(Twine("Failed to open output file: ")
                                          .concat(output_file_path)
                                          .concat("\n")
                                          .str());
  }

  for (const gematria::RegisterAndValue& register_and_value :
       annotated_block.accessed_addrs.initial_regs) {
    std::string register_value_string =
        gematria::ConvertHexToString(register_and_value.register_value);
    output_file << kRegDefPrefix
                << reg_info.getName(register_and_value.register_index) << " "
                << register_value_string << "\n";
  }

  // Multiple mappings can point to the same definition.
  if (annotated_block.accessed_addrs.accessed_blocks.size() > 0) {
    // Make sure to left-pad the memory value string so llvm-exegesis is able to
    // assume the right bit-width.
    std::string memory_value_string =
        absl::StrFormat("%016x", annotated_block.accessed_addrs.block_contents);
    output_file << kMemDefPrefix << kMemNamePrefix << " "
                << annotated_block.accessed_addrs.block_size << " "
                << memory_value_string << "\n";
  }
  for (const auto& addr : annotated_block.accessed_addrs.accessed_blocks) {
    output_file << kMemMapPrefix << kMemNamePrefix << " " << std::dec << addr
                << "\n";
  }

  // Write the loop register annotation, assuming we were able to find one.
  if (annotated_block.loop_register) {
    output_file << kLoopRegisterPrefix
                << reg_info.getName(*annotated_block.loop_register) << "\n";
  }

  // Append disassembled instructions.
  for (const auto& instr :
       annotated_block.basic_block_proto.machine_instructions()) {
    output_file << instr.assembly() << "\n";
  }

  return absl::OkStatus();
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const std::string bhive_filename = absl::GetFlag(FLAGS_bhive_csv);
  if (bhive_filename.empty()) {
    std::cerr << "Error: --bhive_csv is required\n";
    return 1;
  }

  const std::string json_output_dir = absl::GetFlag(FLAGS_json_output_dir);
  const std::string asm_output_dir = absl::GetFlag(FLAGS_asm_output_dir);

  const unsigned blocks_per_json_file =
      absl::GetFlag(FLAGS_blocks_per_json_file);
  if (blocks_per_json_file <= 0) {
    std::cerr << "Error: --blocks_per_json_file must be greater than 1.\n";
    return 1;
  }

  const AnnotatorType annotator_implementation =
      absl::GetFlag(FLAGS_annotator_implementation);

  const std::unique_ptr<gematria::LlvmArchitectureSupport> llvm_support =
      gematria::LlvmArchitectureSupport::X86_64();
  const llvm::MCRegisterInfo& reg_info = llvm_support->mc_register_info();

  gematria::X86Canonicalizer canonicalizer(&llvm_support->target_machine());
  gematria::BHiveImporter bhive_importer(&canonicalizer);

  llvm::exegesis::InitializeX86ExegesisTarget();

  auto llvm_state_or_error = llvm::exegesis::LLVMState::Create("", "native");
  if (!llvm_state_or_error) {
    std::cerr << "Failed to create LLVMState\n";
    return 1;
  }

  std::unique_ptr<gematria::ExegesisAnnotator> exegesis_annotator = nullptr;
  if (annotator_implementation == AnnotatorType::kExegesis) {
    auto exegesis_annotator_or_error =
        gematria::ExegesisAnnotator::create(*llvm_state_or_error);
    if (!exegesis_annotator_or_error) {
      std::cerr << "Failed to create exegesis annotator\n";
      return 1;
    }
    exegesis_annotator = std::move(*exegesis_annotator_or_error);
  }

  std::unique_ptr<llvm::MCInstPrinter> inst_printer =
      llvm_support->CreateMCInstPrinter(0);

  std::ifstream bhive_csv_file(bhive_filename);
  llvm::json::Array processed_snippets;
  const unsigned max_bb_count = absl::GetFlag(FLAGS_max_bb_count);
  const unsigned report_progress_every =
      absl::GetFlag(FLAGS_report_progress_every);
  const bool skip_no_loop_register = absl::GetFlag(FLAGS_skip_no_loop_register);
  const unsigned max_annotation_attempts =
      absl::GetFlag(FLAGS_max_annotation_attempts);
  unsigned int file_counter = 0;
  unsigned int loop_register_failures = 0;
  for (std::string line; std::getline(bhive_csv_file, line);) {
    if (file_counter >= max_bb_count) break;

    auto comma_index = line.find(',');
    if (comma_index == std::string::npos) {
      std::cerr << "Invalid CSV file: no comma in line '" << line << "'\n";
      return 2;
    }

    std::string_view hex = std::string_view(line).substr(0, comma_index);

    auto annotated_block = AnnotateBasicBlock(
        hex, bhive_importer, exegesis_annotator.get(), *llvm_support,
        *inst_printer, max_annotation_attempts);

    if (!annotated_block.ok()) {
      std::cerr << "Failed to annotate block: " << annotated_block.status()
                << "\n";
      return 2;
    }

    // If we can't find a loop register, skip writing out this basic block
    // so that downstream tooling doesn't execute the incorrect number of
    // iterations.
    if (!annotated_block->loop_register && skip_no_loop_register) {
      std::cerr
          << "Skipping block due to not being able to find a loop register\n";
      ++loop_register_failures;
      continue;
    }

    if (!asm_output_dir.empty()) {
      absl::Status asm_output_error = WriteAsmOutput(
          *annotated_block, asm_output_dir, file_counter, reg_info);
      if (!asm_output_error.ok()) {
        std::cerr << "Failed to write block to file: " << asm_output_error
                  << "\n";
        return 2;
      }
    }

    if (!json_output_dir.empty()) {
      processed_snippets.push_back(GetJSONForSnippet(*annotated_block, hex));

      if ((file_counter + 1) % blocks_per_json_file == 0) {
        size_t json_file_number = file_counter / blocks_per_json_file;
        bool write_successfully = WriteJsonFile(
            std::move(processed_snippets), json_file_number, json_output_dir);
        if (!write_successfully) return 4;
        processed_snippets.clear();
      }
    }

    if (file_counter != 0 && file_counter % report_progress_every == 0)
      std::cerr << "Finished annotating block #" << file_counter << ".\n";

    file_counter++;
  }

  if (!json_output_dir.empty() && processed_snippets.size() != 0) {
    size_t json_file_number = file_counter / blocks_per_json_file;
    bool write_successfully = WriteJsonFile(std::move(processed_snippets),
                                            json_file_number, json_output_dir);
    if (!write_successfully) return 4;
  }

  std::cerr << "Failed to find a loop register for " << loop_register_failures
            << " blocks\n";

  return 0;
}
