## Check that llvm-cm recognizes an invalid csv file even if the
## format is valid.
# RUN: split-file %s %t
# RUN: llvm-mc -o %t.o --filetype=obj -triple=x86_64-unknown-linux-gnu %t/test.s
# RUN: not llvm-cm %t.o --csv=%t/mismatched-input-func.csv -granite_model=%S/Inputs/gb-token-mit-2022_12_02.tflite -evaluator=count 2>&1 | FileCheck %t/test.s
# RUN: not llvm-cm %t.o --csv=%t/mismatched-bb.csv -granite_model=%S/Inputs/gb-token-mit-2022_12_02.tflite -evaluator=granite 2>&1 | FileCheck %t/test.s --check-prefix=CHECK-BBERR

//--- mismatched-bb.csv
multiply,1792,1.000000e+00

//--- mismatched-input-func.csv
main,0,1.000000e+00

//--- test.s
# CHECK: error: Function multiply not found in CSV file
# CHECK-BBERR: error: Basic block index not found in CSV file
 .text
 .file "test.c"
 .globl multiply                        # -- Begin function multiply
 .p2align 4, 0x90
 .type multiply,@function
multiply:                               # @multiply
.Lfunc_begin0:
 .cfi_startproc
# %bb.0:
 movl %edi, %eax
 imull %esi, %eax
 retq
.LBB_END0_0:
.Lfunc_end0:
 .size multiply, .Lfunc_end0-multiply
 .cfi_endproc
 .section .llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text
 .byte 2                               # version
 .byte 0                               # feature
 .quad .Lfunc_begin0                   # function address
 .byte 1                               # number of basic blocks
 .byte 0                               # BB id
 .uleb128 .Lfunc_begin0-.Lfunc_begin0
 .uleb128 .LBB_END0_0-.Lfunc_begin0
 .byte 1
 .text
                                        # -- End function
 .globl abs_val                         # -- Begin function abs_val
 .p2align 4, 0x90
 .type abs_val,@function
abs_val:                                # @abs_val
.Lfunc_begin1:
 .cfi_startproc
# %bb.0:
 movl %edi, %eax
 negl %eax
 cmovsl %edi, %eax
 retq
.LBB_END1_0:
.Lfunc_end1:
 .size abs_val, .Lfunc_end1-abs_val
 .cfi_endproc
 .section .llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text
 .byte 2                               # version
 .byte 0                               # feature
 .quad .Lfunc_begin1                   # function address
 .byte 1                               # number of basic blocks
 .byte 0                               # BB id
 .uleb128 .Lfunc_begin1-.Lfunc_begin1
 .uleb128 .LBB_END1_0-.Lfunc_begin1
 .byte 1
 .text
                                        # -- End function
 .ident "Debian clang version 14.0.6"
 .section ".note.GNU-stack","",@progbits
