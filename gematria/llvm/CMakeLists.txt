include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/X86
  ${LLVM_BINARY_DIR}/lib/Target/X86
)

set(LLVM_LINK_COMPONENTS
  CodeGen
  CodeGenTypes
  Core
  MCA
  MC
  MCParser
  Support
  TargetParser
  X86
)

add_llvm_library(GematriaLLVM
  canonicalizer.cc
  disassembler.cc
  llvm_architecture_support.cc

  DEPENDS
  intrinsics_gen
  X86CommonTableGen
)
