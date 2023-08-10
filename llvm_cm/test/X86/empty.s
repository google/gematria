## Check that llvm-cm does not produce any output on an empty input file.
# RUN: llvm-mc -o %t.o --filetype=obj -triple=x86_64-unknown-linux-gnu %s
# RUN: llvm-cm %t.o -csv=%p/Inputs/dummy.csv -granite_model=%S/Inputs/gb-token-mit-2022_12_02.tflite| count 0

main:
