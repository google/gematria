cmake_minimum_required(VERSION 3.20.0)

include_directories(.)
include_directories(${LLVM_SOURCE_DIR})

add_subdirectory(gematria)
add_subdirectory(llvm_cm)
