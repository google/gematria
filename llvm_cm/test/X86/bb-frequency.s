## LLVM-CM frequency weighting test.
# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -o %t.o --filetype=obj -triple=x86_64-unknown-linux-gnu %t/bb-frequency-test.s
# RUN: llvm-cm %t.o --csv=%t/bb-frequency.csv -granite_model=%S/Inputs/gb-token-mit-2022_12_02.tflite -evaluator=granite| FileCheck %t/bb-frequency-test.s
# RUN: llvm-cm %t.o --csv=%t/bb-frequency.csv -granite_model=%S/Inputs/gb-token-mit-2022_12_02.tflite -evaluator=count| FileCheck %t/bb-frequency-test.s --check-prefix=CHECK-COUNT

//--- bb-frequency.csv
main,0,1.000000e+00
main,2,6.190476e-01
main,3,3.809524e-01


//--- bb-frequency-test.s
# CHECK:      <main>:
# CHECK-NEXT: Calculated Frequency: 1.803670e+02

# CHECK-COUNT: <main>:
# CHECK-COUNT: Calculated Frequency: 6.000000e+00

 .text
 .file "bb-frequency.ll"
 .globl main                            # -- Begin function main
 .p2align 4, 0x90
 .type main,@function
main:                                   # @main
.Lfunc_begin0:
 .cfi_startproc
# %bb.0:                                # %entry
 movl %edi, %eax
 addl %esi, %eax
 testl %eax, %eax
 jle .LBB0_2
.LBB_END0_0:
.LBB0_1:                                # %is_pos_true
 addl $10, %eax
 retq
.LBB_END0_1:
.LBB0_2:                                # %is_pos_false
 addl $-10, %eax
 retq
.LBB_END0_2:
.Lfunc_end0:
 .size main, .Lfunc_end0-main
 .cfi_endproc
 .section .llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text
 .byte 2                               # version
 .byte 0                               # feature
 .quad .Lfunc_begin0                   # function address
 .byte 3                               # number of basic blocks
 .byte 0                               # BB id
 .uleb128 .Lfunc_begin0-.Lfunc_begin0
 .uleb128 .LBB_END0_0-.Lfunc_begin0
 .byte 8
 .byte 2                               # BB id
 .uleb128 .LBB0_1-.LBB_END0_0
 .uleb128 .LBB_END0_1-.LBB0_1
 .byte 1
 .byte 3                               # BB id
 .uleb128 .LBB0_2-.LBB_END0_1
 .uleb128 .LBB_END0_2-.LBB0_2
 .byte 1
 .text
                                        # -- End function
 .section ".note.GNU-stack","",@progbits
