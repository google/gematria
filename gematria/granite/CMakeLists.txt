add_llvm_library(GematriaGraphBuilder
  graph_builder.cc
  graph_builder_model_inference.cc

  LINK_LIBS
  tensorflow-lite::tensorflow-lite
)

add_llvm_tool(llvm-granite
  graph_builder_model_inference_main.cc
)

target_link_libraries(llvm-granite PRIVATE
  GematriaBasicBlock
  GematriaGraphBuilder
  GematriaLLVM
  GematriaTFOps
  GematriaUtils
)
