// Copyright 2024 Google Inc.
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

#include "extract_bbs_from_obj_lib.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

using namespace llvm;

namespace gematria {

Expected<std::vector<std::string>> getBasicBlockHexValues(
    MemoryBufferRef binary_memory_buffer) {
  Expected<std::unique_ptr<object::Binary>> ObjBinaryOrErr =
      object::createBinary(binary_memory_buffer);
  if (!ObjBinaryOrErr) {
    return ObjBinaryOrErr.takeError();
  }

  object::Binary &Binary = **ObjBinaryOrErr;
  object::ObjectFile *Obj = cast<object::ObjectFile>(&Binary);

  std::vector<std::string> BasicBlockHexValues;

  for (const auto &Section : Obj->sections()) {
    if (!Section.isText()) continue;

    DenseMap<uint64_t, object::BBAddrMap> BBAddrMap;
    if (const auto *Elf = dyn_cast<object::ELFObjectFileBase>(Obj)) {
      Expected<std::vector<object::BBAddrMap>> BBAddrMappingOrErr =
          Elf->readBBAddrMap(Section.getIndex());
      if (!BBAddrMappingOrErr) {
        return BBAddrMappingOrErr.takeError();
      }

      for (auto &BBAddr : *BBAddrMappingOrErr) {
        BBAddrMap.try_emplace(BBAddr.getFunctionAddress(), std::move(BBAddr));
      }
    } else {
      return make_error<StringError>(errc::invalid_argument,
                                     "Specified object file is not ELF.");
    }

    std::vector<std::pair<uint64_t, uint64_t>> BasicBlocks;

    for (const auto &[FunctionAddress, BasicBlockAddressMap] : BBAddrMap) {
      for (const auto &BasicBlockEntry : BasicBlockAddressMap.getBBEntries()) {
        uint64_t StartAddress = FunctionAddress + BasicBlockEntry.Offset;
        BasicBlocks.push_back(
            std::make_pair(StartAddress, BasicBlockEntry.Size));
      }
    }

    // Sort the basic blocks by start address as we assume this holds later
    // on when iterating through all the basic blocks.
    std::sort(BasicBlocks.begin(), BasicBlocks.end(), [](auto &LHS, auto &RHS) {
      return std::get<0>(LHS) < std::get<0>(RHS);
    });

    if (BasicBlocks.size() == 0) {
      dbgs() << "No basic blocks present in section.\n";
      continue;
    }

    size_t BasicBlockIndex = 0;

    uint64_t SectionEndAddress = Section.getAddress() + Section.getSize();

    uint64_t CurrentAddress = std::get<0>(BasicBlocks[BasicBlockIndex]);

    if (SectionEndAddress < CurrentAddress) continue;

    Expected<StringRef> SectionContentsOrErr = Section.getContents();
    if (!SectionContentsOrErr) {
      return SectionContentsOrErr.takeError();
    }

    while (CurrentAddress < SectionEndAddress &&
           BasicBlockIndex < BasicBlocks.size()) {
      uint64_t OffsetInSection = CurrentAddress - Section.getAddress();
      StringRef BasicBlock(SectionContentsOrErr->data() + OffsetInSection,
                           std::get<1>(BasicBlocks[BasicBlockIndex]));
      std::string BBHex = toHex(BasicBlock);
      BasicBlockHexValues.push_back(std::move(BBHex));

      BasicBlockIndex++;
      if (BasicBlockIndex >= BasicBlocks.size()) {
        break;
      }
      CurrentAddress = std::get<0>(BasicBlocks[BasicBlockIndex]);
    }
  }

  return BasicBlockHexValues;
}

}  // namespace gematria
