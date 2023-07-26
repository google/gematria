## Check that llvm-cm fails with an error when given an invalid triple.
# RUN: llvm-mc -o %t.o --filetype=obj -triple=x86_64-unknown-linux-gnu %s
# RUN: not llvm-cm -triple=not_real_triple %t.o --csv=%p/Inputs/dummy.csv 2>&1 | FileCheck %s

# CHECK: llvm-cm: error: No available targets are compatible with triple "not_real_triple"
