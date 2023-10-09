set(LLVM_LINK_COMPONENTS
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
)

add_llvm_tool(llvm-cm
  llvm-cm.cpp
)

target_link_libraries(llvm-cm PRIVATE
  GematriaGraphBuilder
  GematriaLLVM
  GematriaBasicBlock
  GematriaTFOps
  GematriaUtils
)
