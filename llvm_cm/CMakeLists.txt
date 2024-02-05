add_subdirectory(tools)

configure_lit_site_cfg(
  "${CMAKE_CURRENT_SOURCE_DIR}/test/lit.site.cfg.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg"
)

add_lit_testsuite(check-llvm-tools-llvm-cm "Running llvm-cm tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "llvm-cm" "yaml2obj" "not" "FileCheck" "count" "split-file" "llvm-mc"
)
