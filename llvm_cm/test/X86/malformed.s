## Check that llvm-cm returns an error when run on a non-object file.
# RUN: not llvm-cm %s --csv=%p/Inputs/dummy.csv 2>&1 | FileCheck %s

# CHECK: error: reading file: The file was not recognized as a valid object file
