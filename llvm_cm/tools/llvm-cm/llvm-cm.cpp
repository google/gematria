//===- llvm-cm.cpp - LLVM cost modeling tool
//----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// llvm-cm is a tool for native cost model evaluation.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

// Define the command line options.
static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"), cl::Required);
static cl::opt<std::string> TripleName("triple",
                                       cl::desc("Target triple name."),
                                       cl::init(LLVM_DEFAULT_TARGET_TRIPLE),
                                       cl::value_desc("triple"));

static cl::opt<StringRef> CPU("mcpu", cl::desc("Target a specific cpu type."),
                              cl::init("skylake"), cl::value_desc("cpu-name"));

static cl::opt<std::string> CSVFilename(
    "csv",
    cl::desc("CSV file name, for basic block frequencies. llvm-cm requires "
             "profile information as a csv file."),
    cl::value_desc("filename"), cl::Required);

// BB indices in the BBFreqMap that are not present in the CSV file will be
// assigned an "BBFreq::Invalid (-1)" value.
class BBFreq final {
  double Value = Invalid;

 public:
  BBFreq() = default;

  BBFreq(double Val) : Value(Val) {}

  operator double() const { return Value; }

  static constexpr double Invalid = -1;
};

static void exitIf(bool Cond, Twine Message) {
  if (Cond) {
    WithColor::error(errs(), "llvm-cm") << Message << "\n";
    std::exit(1);
  }
}

static SectionFilter getToolSectionFilter(object::ObjectFile const &O,
                                          uint64_t *Idx) {
  // Set the initial index to max so that the first increment will set it to 0.
  if (Idx != nullptr) *Idx = std::numeric_limits<uint64_t>::max();
  return llvm::object::SectionFilter(
      /*Pred=*/
      [Idx](object::SectionRef S) {
        if (Idx != nullptr) *Idx += 1;
        return true;
      },
      /*Obj=*/O);
}

[[noreturn]] static void error(Error Err) {
  logAllUnhandledErrors(std::move(Err), WithColor::error(outs()),
                        "reading file: ");
  outs().flush();
  exit(1);
}

template <typename T>
T unwrapOrError(Expected<T> EO) {
  if (!EO) error(EO.takeError());
  return std::move(*EO);
}

// TODO: Share this with llvm-objdump.cpp.
static uint8_t getElfSymbolType(const llvm::object::ObjectFile &Obj,
                                const llvm::object::SymbolRef &Sym) {
  assert(Obj.isELF());
  if (auto *Elf32LEObj = dyn_cast<llvm::object::ELF32LEObjectFile>(&Obj))
    return unwrapOrError(Elf32LEObj->getSymbol(Sym.getRawDataRefImpl()))
        ->getType();
  if (auto *Elf64LEObj = dyn_cast<llvm::object::ELF64LEObjectFile>(&Obj))
    return unwrapOrError(Elf64LEObj->getSymbol(Sym.getRawDataRefImpl()))
        ->getType();
  if (auto *Elf32BEObj = dyn_cast<llvm::object::ELF32BEObjectFile>(&Obj))
    return unwrapOrError(Elf32BEObj->getSymbol(Sym.getRawDataRefImpl()))
        ->getType();
  if (auto *Elf64BEObj = cast<llvm::object::ELF64BEObjectFile>(&Obj))
    return unwrapOrError(Elf64BEObj->getSymbol(Sym.getRawDataRefImpl()))
        ->getType();
  llvm_unreachable("Unsupported binary format");
}

// TODO: Share this with llvm-objdump.cpp.
SymbolInfoTy createSymbolInfo(const object::ObjectFile &Obj,
                              const object::SymbolRef Symbol) {
  const uint64_t Addr = unwrapOrError(Symbol.getAddress());
  const StringRef SymName = unwrapOrError(Symbol.getName());
  return SymbolInfoTy(Addr, SymName,
                      Obj.isELF() ? getElfSymbolType(Obj, Symbol)
                                  : static_cast<uint8_t>(ELF::STT_NOTYPE));
}

void printFunctionNames(ArrayRef<SymbolInfoTy> Aliases) {
  for (const auto &Alias : Aliases) outs() << "<" << Alias.Name << ">: \n";
}

static void collectBBtoAddressLabels(
    const DenseMap<uint64_t, llvm::object::BBAddrMap> &AddrToBBAddrMap,
    uint64_t SectionAddr, uint64_t Start, uint64_t End,
    std::unordered_map<uint64_t, std::vector<uint64_t>> &Labels) {
  if (AddrToBBAddrMap.empty()) return;
  Labels.clear();
  const uint64_t StartAddress = SectionAddr + Start;
  const uint64_t EndAddress = SectionAddr + End;
  auto Iter = AddrToBBAddrMap.find(StartAddress);
  if (Iter == AddrToBBAddrMap.end()) return;
  for (const llvm::object::BBAddrMap::BBEntry &BB : Iter->second.BBEntries) {
    const uint64_t BBAddress = BB.Offset + Iter->second.Addr;
    if (BBAddress >= EndAddress) continue;
    Labels[BBAddress].push_back(BB.ID);
  }
}

double calcFrequency(uint64_t NumInstsIn, StringRef CurrSymbol,
                     const StringMap<SmallVector<BBFreq, 20>> &BBFreqMap,
                     uint64_t BB) {
  exitIf(!BBFreqMap.contains(CurrSymbol),
         "Function " + CurrSymbol + " not found in CSV file");
  exitIf(BB >= BBFreqMap.lookup(CurrSymbol).size(),
         "Basic block index not found in CSV file: Index " + Twine(BB) +
             " is+ out of bounds");
  exitIf(BBFreqMap.lookup(CurrSymbol)[BB] == BBFreq::Invalid,
         "Basic block index not found in CSV file for function " + CurrSymbol +
             ": Index " + Twine(BB) + " is not present");
  double Freq = BBFreqMap.lookup(CurrSymbol)[BB] * NumInstsIn;
  return Freq;
}

void processInsts(
    MCDisassembler &DisAsm, uint64_t SectionAddr, ArrayRef<uint8_t> Bytes,
    raw_svector_ostream &CommentStream, uint64_t Start, uint64_t End,
    uint64_t Index,
    const std::unordered_map<uint64_t, std::vector<uint64_t>> &Labels,
    StringRef CurrSymbol, const StringMap<SmallVector<BBFreq, 20>> &BBFreqMap) {
  uint64_t NumInstsInBB = 0;
  uint64_t ThisBb = -1;
  bool EnteredBb = false;
  while (Index < End) {
    uint64_t CurrAddr = SectionAddr + Index;
    auto FirstIter = Labels.find(CurrAddr);
    if (FirstIter != Labels.end()) {
      for (auto Label : FirstIter->second) {
        if (EnteredBb) {
          outs() << "Calculated frequency: "
                 << calcFrequency(NumInstsInBB, CurrSymbol, BBFreqMap, ThisBb)
                 << "\n";
          NumInstsInBB = 0;
        }
        EnteredBb = true;
        ThisBb = Label;
        outs() << "<"
               << "BB" + Twine(Label) << ">: ";
        outs() << format("%016" PRIx64 " ", CurrAddr) << "\n";
      }
    }
    MCInst Inst;
    uint64_t Size = 0;
    ArrayRef<uint8_t> BytesSlice = Bytes.slice(Index);
    exitIf(
        !DisAsm.getInstruction(Inst, Size, BytesSlice, CurrAddr, CommentStream),
        "disassembler cannot disassemble given data at address 0x" +
            Twine::utohexstr(CurrAddr).str());
    ++NumInstsInBB;
    if (Size == 0) {
      Size = std::min<uint64_t>(
          BytesSlice.size(), DisAsm.suggestBytesToSkip(BytesSlice, CurrAddr));
    }
    Index += Size;
  }

  if (EnteredBb && Labels.size() > 1) {
    outs() << "Calculated frequency: "
           << calcFrequency(NumInstsInBB, CurrSymbol, BBFreqMap, ThisBb)
           << "\n";
    NumInstsInBB = 0;
    EnteredBb = false;
  } else {
    outs() << "Calculated frequency: "
           << calcFrequency(NumInstsInBB, CurrSymbol, BBFreqMap, ThisBb)
           << "\n";
  }
}

void populateBBFreqMap(StringMap<SmallVector<BBFreq, 20>> &BBFreqMap) {
  if (CSVFilename.empty()) return;

  llvm::ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(CSVFilename);
  exitIf(!FileOrErr, "failed to open file " + CSVFilename);

  constexpr int FuncIdx = 0;
  constexpr int BBIdx = 1;
  constexpr int FreqIdx = 2;

  llvm::line_iterator LineIter(**FileOrErr, /*SkipBlanks=*/true);

  for (; !LineIter.is_at_eof(); ++LineIter) {
    SmallVector<StringRef, 4> Row;
    LineIter->split(Row, ',', 3);
    StringRef FuncName = Row[FuncIdx];
    exitIf(FuncName.empty(), "Function name cannot be empty");
    double FreqVal = BBFreq::Invalid;
    exitIf(Row[FreqIdx].getAsDouble(FreqVal, true),
           "Frequency value could not be parsed");
    uint64_t BBIndex = -1;
    exitIf(Row[BBIdx].getAsInteger(10, BBIndex), "BBIndex could not be parsed");
    auto &VecName = BBFreqMap[FuncName];
    if (BBIndex >= VecName.size()) {
      VecName.resize(BBIndex + 1);
    }
    VecName[BBIndex] = FreqVal;
  }
}

int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "llvm cost model tool\n");

  // Set up the triple and target features.
  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllDisassemblers();

  object::OwningBinary<object::Binary> ObjBinary =
      unwrapOrError(object::createBinary(InputFilename));
  object::Binary &Binary = *ObjBinary.getBinary();
  object::ObjectFile *Obj = cast<object::ObjectFile>(&Binary);

  // Start setting up the disassembler.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
  exitIf(!TheTarget, Error);

  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TripleName));
  assert(MRI && "Unable to create target register info!");

  MCTargetOptions MCOptions;

  std::unique_ptr<MCAsmInfo> AsmInfo(
      TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
  assert(AsmInfo && "Unable to create target asm info");

  Expected<SubtargetFeatures> FeatureVals = Obj->getFeatures();
  assert(FeatureVals && "Could not read features");
  std::unique_ptr<MCSubtargetInfo> SubInfo(TheTarget->createMCSubtargetInfo(
      TripleName, CPU, FeatureVals->getString()));
  assert(SubInfo && "Unable to create target subtarget info!");

  std::unique_ptr<MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  assert(MII && "Unable to create target instruction info!");

  MCContext Ctx(Triple(TripleName), AsmInfo.get(), MRI.get(), SubInfo.get());

  std::unique_ptr<MCObjectFileInfo> MOFI(
      TheTarget->createMCObjectFileInfo(Ctx, false));
  Ctx.setObjectFileInfo(MOFI.get());

  std::unique_ptr<MCDisassembler> DisAsm(
      TheTarget->createMCDisassembler(*SubInfo, Ctx));
  assert(DisAsm && "Unable to create disassembler!");

  StringMap<SmallVector<BBFreq, 20>> BBFreqMap;
  populateBBFreqMap(BBFreqMap);

  // Section information should be stored to determine whether
  // or not the section is relevant to disassembly.
  MapVector<SectionRef, SectionSymbolsTy> AllSymbols;
  SectionSymbolsTy UndefinedSymbols;
  for (const object::SymbolRef &Symbol : Obj->symbols()) {
    auto TypeOrErr = Symbol.getType();
    exitIf(!TypeOrErr, "failed to get symbol type");
    if (TypeOrErr.get() != SymbolRef::ST_Function) continue;
    Expected<StringRef> NameOrErr = Symbol.getName();
    exitIf(!NameOrErr, "failed to get symbol name");

    // If the symbol is a section symbol, then ignore it.
    if (Obj->isELF() && getElfSymbolType(*Obj, Symbol) == ELF::STT_SECTION)
      continue;

    object::section_iterator SectionI = unwrapOrError(Symbol.getSection());

    // If the section iterator does not point to the end of the section
    // list, then the symbol is defined in a section.
    if (SectionI != Obj->section_end()) {
      AllSymbols[*SectionI].push_back(createSymbolInfo(*Obj, Symbol));
    } else {
      UndefinedSymbols.push_back(createSymbolInfo(*Obj, Symbol));
    }
  }

  // Sort the symbols in order of address.
  for (std::pair<SectionRef, SectionSymbolsTy> &SortSymbols : AllSymbols)
    llvm::stable_sort(SortSymbols.second);
  llvm::stable_sort(UndefinedSymbols);

  DenseMap<uint64_t, BBAddrMap> BBAddrMap;
  auto GetBBAddrMapping = [&]() {
    BBAddrMap.clear();
    if (const auto *Elf = dyn_cast<object::ELFObjectFileBase>(Obj)) {
      auto BBAddrMappingOrErr = Elf->readBBAddrMap();
      exitIf(!BBAddrMappingOrErr, "failed to read basic block address mapping");
      for (auto &BBAddr : *BBAddrMappingOrErr) {
        BBAddrMap.try_emplace(BBAddr.Addr, std::move(BBAddr));
      }
    }
  };

  GetBBAddrMapping();

  // Begin iterating over the sections. For each section, get the symbols,
  // instructions and basic blocks and calculate the weighted
  // frequency of each basic block.
  for (const object::SectionRef &Section :
       getToolSectionFilter(*Obj, nullptr)) {
    if ((!Section.isText() || Section.isVirtual())) continue;
    const uint64_t SectionAddr = Section.getAddress();
    const uint64_t SectionSize = Section.getSize();
    if (!SectionSize) continue;
    // Get all the symbols in the section - these were sorted earlier.
    SectionSymbolsTy &SortedSymbols = AllSymbols[Section];

    ArrayRef<uint8_t> Bytes =
        arrayRefFromStringRef(unwrapOrError(Section.getContents()));
    SmallString<40> Comments;
    raw_svector_ostream CommentStream(Comments);

    // For each symbol in the current section, disassemble the instructions
    // and obtain the location of each basic block.
    for (size_t SI = 0, SE = SortedSymbols.size(); SI != SE;) {
      // Find all symbols in the same "location" by incrementing over
      // SI until the starting address changes. The sorted symbols were sorted
      // by address.
      const size_t FirstSI = SI;
      uint64_t Start = SortedSymbols[SI].Addr;

      // If the current symbol's address is the same as the previous
      // symbol's address, then we know that the current symbol is an
      // alias, and we skip it.
      while (SI != SE && SortedSymbols[SI].Addr == Start) ++SI;

      // End is the end of the current location, the start of the next symbol.
      uint64_t End =
          SI < SE ? SortedSymbols[SI].Addr : SectionAddr + SectionSize;

      // The aliases are the symbols that have the same address.
      ArrayRef<SymbolInfoTy> Aliases(&SortedSymbols[FirstSI], SI - FirstSI);

      uint64_t StartAddr = 0;
      // If the symbol range does not overlap with our section,
      // move to the next symbol.
      if (Start >= End || End <= StartAddr) continue;

      // Adjust the start and end addresses to be relative to the start of the
      // section.
      Start -= SectionAddr;
      End -= SectionAddr;

      std::unordered_map<uint64_t, std::vector<uint64_t>> BBtoAddressLabels;
      collectBBtoAddressLabels(BBAddrMap, SectionAddr, Start, End,
                               BBtoAddressLabels);

      printFunctionNames(Aliases);

      uint64_t Index = Start;
      if (SectionAddr < StartAddr)
        Index = std::max<uint64_t>(Index, StartAddr - SectionAddr);

      StringRef CurrSymbolName = Aliases[0].Name;
      processInsts(*DisAsm, SectionAddr, Bytes, CommentStream, Start, End,
                   Index, BBtoAddressLabels, CurrSymbolName, BBFreqMap);
    }
  }
}
