add_llvm_library(GematriaTFOps
  unsorted_segment_sum_op.cc

  LINK_LIBS
  tensorflow-lite::tensorflow-lite
)
