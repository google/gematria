# RUN: split-file %s %t
# RUN: llvm-mc -o %t.o --filetype=obj -triple=x86_64-unknown-linux-gnu %t/profile-dump-test.s
# RUN: llvm-cm %t.o --csv=%t/profile-dump-test.csv -granite_model=%S/Inputs/gb-token-mit-2022_12_02.tflite -evaluator=granite | FileCheck %t/profile-dump-test.s
# RUN: llvm-cm %t.o --csv=%t/profile-dump-test.csv -granite_model=%S/Inputs/gb-token-mit-2022_12_02.tflite -evaluator=count | FileCheck %t/profile-dump-test.s --check-prefix=CHECK-COUNT

//--- profile-dump-test.csv
f2,0,1.000000e+00
f1,0,1.000000e+00
f1,1,5.000000e-01
f1,2,1.000000e+00


//--- profile-dump-test.s
# CHECK:      <f2>:
# CHECK-NEXT: Calculated Frequency: 3.{{[0-9]+}}e+01
# CHECK-NEXT: <f1>:
# CHECK-NEXT: Calculated Frequency: 1.{{[0-9]+}}e+02

# CHECK-COUNT: <f2>:
# CHECK-COUNT: Calculated Frequency: 1.{{[0-9]+}}e+01
# CHECK-COUNT: <f1>:
# CHECK-COUNT: Calculated Frequency: 8.{{[0-9]+}}e+00


 .text
 .file "profile_dump_test.ll"
 .globl f2                              # -- Begin function f2
 .p2align 4, 0x90
 .type f2,@function
f2:                                     # @f2
.Lfunc_begin0:
 .cfi_startproc
# %bb.0:
 leaq (%rdi,%rsi), %rax
 retq
.LBB_END0_0:
.Lfunc_end0:
 .size f2, .Lfunc_end0-f2
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
 .globl f1                              # -- Begin function f1
 .p2align 4, 0x90
 .type f1,@function
f1:                                     # @f1
.Lfunc_begin1:
 .cfi_startproc
# %bb.0:
 pushq %rax
 .cfi_def_cfa_offset 16
 movl $2, %edi
 movl $2, %esi
 callq f2@PLT
 cmpq $4, %rax
 jne .LBB1_2
.LBB_END1_0:
.LBB1_1:                                # %ifEqual
 xorl %eax, %eax
.LBB_END1_1:
.LBB1_2:                                # %ifNotEqual
 popq %rcx
 .cfi_def_cfa_offset 8
 retq
.LBB_END1_2:
.Lfunc_end1:
 .size f1, .Lfunc_end1-f1
 .cfi_endproc
 .section .llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text
 .byte 2                               # version
 .byte 0                               # feature
 .quad .Lfunc_begin1                   # function address
 .byte 3                               # number of basic blocks
 .byte 0                               # BB id
 .uleb128 .Lfunc_begin1-.Lfunc_begin1
 .uleb128 .LBB_END1_0-.Lfunc_begin1
 .byte 8
 .byte 1                               # BB id
 .uleb128 .LBB1_1-.LBB_END1_0
 .uleb128 .LBB_END1_1-.LBB1_1
 .byte 8
 .byte 2                               # BB id
 .uleb128 .LBB1_2-.LBB_END1_1
 .uleb128 .LBB_END1_2-.LBB1_2
 .byte 1
 .text
                                        # -- End function
 .section ".note.GNU-stack","",@progbits
