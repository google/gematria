## Check that llvm-cm can handle input containing many basic blocks across functions.
# RUN: llvm-mc -o %t.o --filetype=obj -triple=x86_64-unknown-linux-gnu %s
# RUN: llvm-cm %t.o -csv=%S/Inputs/multi-func.csv -granite_model=%S/Inputs/gb-token-mit-2022_12_02.tflite -evaluator=granite | FileCheck %s
# RUN: llvm-cm %t.o -csv=%S/Inputs/multi-func.csv -granite_model=%S/Inputs/gb-token-mit-2022_12_02.tflite -evaluator=count | FileCheck %s --check-prefix=CHECK-COUNT


# CHECK:      <reverse>:
# CHECK-NEXT: Calculated Frequency: 8.3{{[0-9]+}}e+03
# CHECK-NEXT: <tallestBillboard>:
# CHECK-NEXT: Calculated Frequency: 2.9{{[0-9]+}}e+05
# CHECK-NEXT: <isMatch>:
# CHECK-NEXT: Calculated Frequency: 8.2{{[0-9]+}}e+02
# CHECK-NEXT: <bubbleSort>:
# CHECK-NEXT: Calculated Frequency: 6.6{{[0-9]+}}e+04
# CHECK-NEXT: <isPrime>:
# CHECK-NEXT: Calculated Frequency: 5.7{{[0-9]+}}e+02
# CHECK-NEXT: <main>:
# CHECK-NEXT: Calculated Frequency: 8.8{{[0-9]+}}e+03


# CHECK-COUNT: <reverse>:
# CHECK-COUNT: Calculated Frequency: 1.4{{[0-9]+}}e+02
# CHECK-COUNT: <tallestBillboard>:
# CHECK-COUNT: Calculated Frequency: 4.4{{[0-9]+}}e+03
# CHECK-COUNT: <isMatch>:
# CHECK-COUNT: Calculated Frequency: 2.2{{[0-9]+}}e+01
# CHECK-COUNT: <bubbleSort>:
# CHECK-COUNT: Calculated Frequency: 2.0{{[0-9]+}}e+03
# CHECK-COUNT: <isPrime>:
# CHECK-COUNT: Calculated Frequency: 2.7{{[0-9]+}}e+01
# CHECK-COUNT: <main>:
# CHECK-COUNT: Calculated Frequency: 2.3{{[0-9]+}}e+02

 .text
 .file "test.c"
 .globl reverse                         # -- Begin function reverse
 .p2align 4, 0x90
 .type reverse,@function
reverse:                                # @reverse
.Lfunc_begin0:
 .cfi_startproc
# %bb.0:
 cmpb $0, (%rdi)
 je .LBB0_4
.LBB_END0_0:
.LBB0_1:                                # %.preheader1
 xorl %ecx, %ecx
.LBB_END0_1:
 .p2align 4, 0x90
.LBB0_2:                                # =>This Inner Loop Header: Depth=1
 cmpb $0, 1(%rdi,%rcx)
 leaq 1(%rcx), %rcx
 jne .LBB0_2
.LBB_END0_2:
.LBB0_3:
 leaq (%rdi,%rcx), %rax
 jmp .LBB0_5
.LBB_END0_3:
.LBB0_4:
 xorl %ecx, %ecx
 movq %rdi, %rax
.LBB_END0_4:
.LBB0_5:
 addq %rdi, %rcx
 decq %rcx
 cmpq %rcx, %rax
 jae .LBB0_9
.LBB_END0_5:
.LBB0_6:
 movzbl (%rcx), %edx
 movb %dl, (%rax)
 movb $0, (%rcx)
 leaq 1(%rax), %rdx
 decq %rcx
 cmpq %rcx, %rdx
 jae .LBB0_9
.LBB_END0_6:
.LBB0_7:                                # %.preheader
 addq $2, %rax
.LBB_END0_7:
 .p2align 4, 0x90
.LBB0_8:                                # =>This Inner Loop Header: Depth=1
 movzbl -1(%rax), %edx
 movzbl (%rcx), %esi
 movb %sil, -1(%rax)
 movb %dl, (%rcx)
 decq %rcx
 leaq 1(%rax), %rdx
 cmpq %rcx, %rax
 movq %rdx, %rax
 jb .LBB0_8
.LBB_END0_8:
.LBB0_9:
 retq
.LBB_END0_9:
.Lfunc_end0:
 .size reverse, .Lfunc_end0-reverse
 .cfi_endproc
 .section .llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text
 .byte 2                               # version
 .byte 0                               # feature
 .quad .Lfunc_begin0                   # function address
 .byte 10                              # number of basic blocks
 .byte 0                               # BB id
 .uleb128 .Lfunc_begin0-.Lfunc_begin0
 .uleb128 .LBB_END0_0-.Lfunc_begin0
 .byte 8
 .byte 1                               # BB id
 .uleb128 .LBB0_1-.LBB_END0_0
 .uleb128 .LBB_END0_1-.LBB0_1
 .byte 8
 .byte 2                               # BB id
 .uleb128 .LBB0_2-.LBB_END0_1
 .uleb128 .LBB_END0_2-.LBB0_2
 .byte 8
 .byte 3                               # BB id
 .uleb128 .LBB0_3-.LBB_END0_2
 .uleb128 .LBB_END0_3-.LBB0_3
 .byte 0
 .byte 9                               # BB id
 .uleb128 .LBB0_4-.LBB_END0_3
 .uleb128 .LBB_END0_4-.LBB0_4
 .byte 8
 .byte 4                               # BB id
 .uleb128 .LBB0_5-.LBB_END0_4
 .uleb128 .LBB_END0_5-.LBB0_5
 .byte 8
 .byte 5                               # BB id
 .uleb128 .LBB0_6-.LBB_END0_5
 .uleb128 .LBB_END0_6-.LBB0_6
 .byte 8
 .byte 6                               # BB id
 .uleb128 .LBB0_7-.LBB_END0_6
 .uleb128 .LBB_END0_7-.LBB0_7
 .byte 8
 .byte 7                               # BB id
 .uleb128 .LBB0_8-.LBB_END0_7
 .uleb128 .LBB_END0_8-.LBB0_8
 .byte 8
 .byte 8                               # BB id
 .uleb128 .LBB0_9-.LBB_END0_8
 .uleb128 .LBB_END0_9-.LBB0_9
 .byte 1
 .text
                                        # -- End function
 .globl tallestBillboard                # -- Begin function tallestBillboard
 .p2align 4, 0x90
 .type tallestBillboard,@function
tallestBillboard:                       # @tallestBillboard
.Lfunc_begin1:
 .cfi_startproc
# %bb.0:
 testl %esi, %esi
 jle .LBB1_3
.LBB_END1_0:
.LBB1_1:
 pushq %rbp
 .cfi_def_cfa_offset 16
 .cfi_offset %rbp, -16
 movq %rsp, %rbp
 .cfi_def_cfa_register %rbp
 pushq %r15
 pushq %r14
 pushq %r13
 pushq %r12
 pushq %rbx
 subq $40, %rsp
 .cfi_offset %rbx, -56
 .cfi_offset %r12, -48
 .cfi_offset %r13, -40
 .cfi_offset %r14, -32
 .cfi_offset %r15, -24
 movl %esi, %r14d
 movl %esi, %r8d
 cmpl $8, %esi
 jae .LBB1_4
.LBB_END1_1:
.LBB1_2:
 xorl %eax, %eax
 xorl %r13d, %r13d
 jmp .LBB1_12
.LBB_END1_2:
.LBB1_3:
 .cfi_def_cfa %rsp, 8
 .cfi_restore %rbx
 .cfi_restore %rbp
 .cfi_restore %r12
 .cfi_restore %r13
 .cfi_restore %r14
 .cfi_restore %r15
 xorl %eax, %eax
 retq
.LBB_END1_3:
.LBB1_4:
 .cfi_def_cfa %rbp, 16
 .cfi_offset %rbx, -56
 .cfi_offset %rbp, -16
 .cfi_offset %r12, -48
 .cfi_offset %r13, -40
 .cfi_offset %r14, -32
 .cfi_offset %r15, -24
 movl %r8d, %eax
 andl $-8, %eax
 leaq -8(%rax), %rdx
 movq %rdx, %rsi
 shrq $3, %rsi
 incq %rsi
 movl %esi, %ecx
 andl $3, %ecx
 cmpq $24, %rdx
 jae .LBB1_6
.LBB_END1_4:
.LBB1_5:
 pxor %xmm0, %xmm0
 xorl %edx, %edx
 pxor %xmm1, %xmm1
 jmp .LBB1_8
.LBB_END1_5:
.LBB1_6:
 andq $-4, %rsi
 pxor %xmm0, %xmm0
 xorl %edx, %edx
 pxor %xmm1, %xmm1
.LBB_END1_6:
 .p2align 4, 0x90
.LBB1_7:                                # =>This Inner Loop Header: Depth=1
 movdqu (%rdi,%rdx,4), %xmm2
 paddd %xmm0, %xmm2
 movdqu 16(%rdi,%rdx,4), %xmm0
 paddd %xmm1, %xmm0
 movdqu 32(%rdi,%rdx,4), %xmm1
 movdqu 48(%rdi,%rdx,4), %xmm3
 movdqu 64(%rdi,%rdx,4), %xmm4
 paddd %xmm1, %xmm4
 paddd %xmm2, %xmm4
 movdqu 80(%rdi,%rdx,4), %xmm2
 paddd %xmm3, %xmm2
 paddd %xmm0, %xmm2
 movdqu 96(%rdi,%rdx,4), %xmm0
 paddd %xmm4, %xmm0
 movdqu 112(%rdi,%rdx,4), %xmm1
 paddd %xmm2, %xmm1
 addq $32, %rdx
 addq $-4, %rsi
 jne .LBB1_7
.LBB_END1_7:
.LBB1_8:
 testq %rcx, %rcx
 je .LBB1_11
.LBB_END1_8:
.LBB1_9:                                # %.preheader
 leaq (%rdi,%rdx,4), %rdx
 addq $16, %rdx
 shlq $5, %rcx
 xorl %esi, %esi
.LBB_END1_9:
 .p2align 4, 0x90
.LBB1_10:                               # =>This Inner Loop Header: Depth=1
 movdqu -16(%rdx,%rsi), %xmm2
 paddd %xmm2, %xmm0
 movdqu (%rdx,%rsi), %xmm2
 paddd %xmm2, %xmm1
 addq $32, %rsi
 cmpq %rsi, %rcx
 jne .LBB1_10
.LBB_END1_10:
.LBB1_11:
 paddd %xmm1, %xmm0
 pshufd $238, %xmm0, %xmm1              # xmm1 = xmm0[2,3,2,3]
 paddd %xmm0, %xmm1
 pshufd $85, %xmm1, %xmm0               # xmm0 = xmm1[1,1,1,1]
 paddd %xmm1, %xmm0
 movd %xmm0, %r13d
 cmpq %r8, %rax
 je .LBB1_13
.LBB_END1_11:
 .p2align 4, 0x90
.LBB1_12:                               # =>This Inner Loop Header: Depth=1
 addl (%rdi,%rax,4), %r13d
 incq %rax
 cmpq %rax, %r8
 jne .LBB1_12
.LBB_END1_12:
.LBB1_13:
 movq %r8, -72(%rbp)                  # 8-byte Spill
 movq %rdi, -80(%rbp)                 # 8-byte Spill
 leal 1(%r13), %eax
 movq %rsp, %rbx
 leaq 15(,%rax,4), %r12
 andq $-16, %r12
 movq %rbx, %r15
 subq %r12, %r15
 movq %r15, %rsp
 movq %r12, -64(%rbp)                 # 8-byte Spill
 negq %r12
 movq %rax, -48(%rbp)                 # 8-byte Spill
 leaq (,%rax,4), %rdx
 movq %r15, %rdi
 movl $255, %esi
 movq %rdx, -56(%rbp)                 # 8-byte Spill
 callq memset@PLT
 movl $0, (%rbx,%r12)
 testl %r14d, %r14d
 jle .LBB1_22
.LBB_END1_13:
.LBB1_14:
 xorl %r12d, %r12d
 jmp .LBB1_16
.LBB_END1_14:
 .p2align 4, 0x90
.LBB1_15:                               #   in Loop: Header=BB1_16 Depth=1
 movq %rbx, %rsp
 incq %r12
 cmpq %r9, %r12
 je .LBB1_21
.LBB_END1_15:
.LBB1_16:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_19 Depth 2
 movq %rsp, %rbx
 movq %rsp, %r14
 subq -64(%rbp), %r14                 # 8-byte Folded Reload
 movq %r14, %rsp
 movq %r14, %rdi
 movq %r15, %rsi
 movq -56(%rbp), %rdx                 # 8-byte Reload
 callq memcpy@PLT
 movq -80(%rbp), %rax                 # 8-byte Reload
 movslq (%rax,%r12,4), %rdx
 cmpl %edx, %r13d
 movq -72(%rbp), %r9                  # 8-byte Reload
 jl .LBB1_15
.LBB_END1_16:
.LBB1_17:                               #   in Loop: Header=BB1_16 Depth=1
 xorps %xmm0, %xmm0
 cvtsi2sd %edx, %xmm0
 movq -48(%rbp), %rax                 # 8-byte Reload
                                        # kill: def $eax killed $eax killed $rax def $rax
 subl %edx, %eax
 movl %edx, %ecx
 negl %ecx
 leaq (%r15,%rdx,4), %rdx
 xorl %esi, %esi
 jmp .LBB1_19
.LBB_END1_17:
 .p2align 4, 0x90
.LBB1_18:                               #   in Loop: Header=BB1_19 Depth=2
 incq %rsi
 cmpq %rsi, %rax
 je .LBB1_15
.LBB_END1_18:
.LBB1_19:                               #   Parent Loop BB1_16 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
 movl (%r14,%rsi,4), %edi
 cmpl $-1, %edi
 je .LBB1_18
.LBB_END1_19:
.LBB1_20:                               #   in Loop: Header=BB1_19 Depth=2
 xorps %xmm1, %xmm1
 cvtsi2sdl (%rdx,%rsi,4), %xmm1
 xorps %xmm2, %xmm2
 cvtsi2sd %edi, %xmm2
 maxsd %xmm2, %xmm1
 cvttsd2si %xmm1, %edi
 movl %edi, (%rdx,%rsi,4)
 leal (%rcx,%rsi), %edi
 movl %edi, %r8d
 negl %r8d
 cmovsl %edi, %r8d
 xorps %xmm1, %xmm1
 cvtsi2sdl (%r15,%r8,4), %xmm1
 xorps %xmm3, %xmm3
 cvtsi2sd %esi, %xmm3
 movapd %xmm0, %xmm4
 minsd %xmm3, %xmm4
 addsd %xmm2, %xmm4
 maxsd %xmm1, %xmm4
 cvttsd2si %xmm4, %edi
 movl %edi, (%r15,%r8,4)
 jmp .LBB1_18
.LBB_END1_20:
.LBB1_21:
 movl (%r15), %eax
 jmp .LBB1_23
.LBB_END1_21:
.LBB1_22:
 xorl %eax, %eax
.LBB_END1_22:
.LBB1_23:
 leaq -40(%rbp), %rsp
 popq %rbx
 popq %r12
 popq %r13
 popq %r14
 popq %r15
 popq %rbp
 .cfi_def_cfa %rsp, 8
 .cfi_restore %rbx
 .cfi_restore %r12
 .cfi_restore %r13
 .cfi_restore %r14
 .cfi_restore %r15
 .cfi_restore %rbp
 retq
.LBB_END1_23:
.Lfunc_end1:
 .size tallestBillboard, .Lfunc_end1-tallestBillboard
 .cfi_endproc
 .section .llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text
 .byte 2                               # version
 .byte 0                               # feature
 .quad .Lfunc_begin1                   # function address
 .byte 24                              # number of basic blocks
 .byte 0                               # BB id
 .uleb128 .Lfunc_begin1-.Lfunc_begin1
 .uleb128 .LBB_END1_0-.Lfunc_begin1
 .byte 8
 .byte 1                               # BB id
 .uleb128 .LBB1_1-.LBB_END1_0
 .uleb128 .LBB_END1_1-.LBB1_1
 .byte 8
 .byte 22                              # BB id
 .uleb128 .LBB1_2-.LBB_END1_1
 .uleb128 .LBB_END1_2-.LBB1_2
 .byte 0
 .byte 21                              # BB id
 .uleb128 .LBB1_3-.LBB_END1_2
 .uleb128 .LBB_END1_3-.LBB1_3
 .byte 1
 .byte 2                               # BB id
 .uleb128 .LBB1_4-.LBB_END1_3
 .uleb128 .LBB_END1_4-.LBB1_4
 .byte 8
 .byte 23                              # BB id
 .uleb128 .LBB1_5-.LBB_END1_4
 .uleb128 .LBB_END1_5-.LBB1_5
 .byte 0
 .byte 3                               # BB id
 .uleb128 .LBB1_6-.LBB_END1_5
 .uleb128 .LBB_END1_6-.LBB1_6
 .byte 8
 .byte 4                               # BB id
 .uleb128 .LBB1_7-.LBB_END1_6
 .uleb128 .LBB_END1_7-.LBB1_7
 .byte 8
 .byte 5                               # BB id
 .uleb128 .LBB1_8-.LBB_END1_7
 .uleb128 .LBB_END1_8-.LBB1_8
 .byte 8
 .byte 6                               # BB id
 .uleb128 .LBB1_9-.LBB_END1_8
 .uleb128 .LBB_END1_9-.LBB1_9
 .byte 8
 .byte 7                               # BB id
 .uleb128 .LBB1_10-.LBB_END1_9
 .uleb128 .LBB_END1_10-.LBB1_10
 .byte 8
 .byte 8                               # BB id
 .uleb128 .LBB1_11-.LBB_END1_10
 .uleb128 .LBB_END1_11-.LBB1_11
 .byte 8
 .byte 12                              # BB id
 .uleb128 .LBB1_12-.LBB_END1_11
 .uleb128 .LBB_END1_12-.LBB1_12
 .byte 8
 .byte 10                              # BB id
 .uleb128 .LBB1_13-.LBB_END1_12
 .uleb128 .LBB_END1_13-.LBB1_13
 .byte 8
 .byte 11                              # BB id
 .uleb128 .LBB1_14-.LBB_END1_13
 .uleb128 .LBB_END1_14-.LBB1_14
 .byte 0
 .byte 17                              # BB id
 .uleb128 .LBB1_15-.LBB_END1_14
 .uleb128 .LBB_END1_15-.LBB1_15
 .byte 8
 .byte 15                              # BB id
 .uleb128 .LBB1_16-.LBB_END1_15
 .uleb128 .LBB_END1_16-.LBB1_16
 .byte 8
 .byte 16                              # BB id
 .uleb128 .LBB1_17-.LBB_END1_16
 .uleb128 .LBB_END1_17-.LBB1_17
 .byte 0
 .byte 20                              # BB id
 .uleb128 .LBB1_18-.LBB_END1_17
 .uleb128 .LBB_END1_18-.LBB1_18
 .byte 8
 .byte 18                              # BB id
 .uleb128 .LBB1_19-.LBB_END1_18
 .uleb128 .LBB_END1_19-.LBB1_19
 .byte 8
 .byte 19                              # BB id
 .uleb128 .LBB1_20-.LBB_END1_19
 .uleb128 .LBB_END1_20-.LBB1_20
 .byte 0
 .byte 13                              # BB id
 .uleb128 .LBB1_21-.LBB_END1_20
 .uleb128 .LBB_END1_21-.LBB1_21
 .byte 0
 .byte 24                              # BB id
 .uleb128 .LBB1_22-.LBB_END1_21
 .uleb128 .LBB_END1_22-.LBB1_22
 .byte 8
 .byte 25                              # BB id
 .uleb128 .LBB1_23-.LBB_END1_22
 .uleb128 .LBB_END1_23-.LBB1_23
 .byte 1
 .text
                                        # -- End function
 .globl isMatch                         # -- Begin function isMatch
 .p2align 4, 0x90
 .type isMatch,@function
isMatch:                                # @isMatch
.Lfunc_begin2:
 .cfi_startproc
# %bb.0:
 movzbl (%rsi), %ecx
 movzbl (%rdi), %edx
 testb %dl, %dl
 sete %al
 testb %cl, %cl
 je .LBB2_11
.LBB_END2_0:
.LBB2_1:
 pushq %rbp
 .cfi_def_cfa_offset 16
 pushq %r14
 .cfi_def_cfa_offset 24
 pushq %rbx
 .cfi_def_cfa_offset 32
 .cfi_offset %rbx, -32
 .cfi_offset %r14, -24
 .cfi_offset %rbp, -16
 movq %rsi, %rbx
 movq %rdi, %r14
 testb %dl, %dl
 je .LBB2_5
.LBB_END2_1:
.LBB2_2:
 cmpb %cl, %dl
 sete %al
 cmpb $46, %cl
 sete %bpl
 orb %al, %bpl
 cmpb $42, 1(%rbx)
 je .LBB2_6
.LBB_END2_2:
.LBB2_3:
 testb %bpl, %bpl
 je .LBB2_9
.LBB_END2_3:
.LBB2_4:
 incq %rbx
 jmp .LBB2_8
.LBB_END2_4:
.LBB2_5:
 xorl %ebp, %ebp
 cmpb $42, 1(%rbx)
 movl $0, %eax
 jne .LBB2_10
.LBB_END2_5:
.LBB2_6:
 leaq 2(%rbx), %rsi
 movq %r14, %rdi
 callq isMatch
 testl %eax, %eax
 setne %al
 jne .LBB2_10
.LBB_END2_6:
.LBB2_7:
 testb %bpl, %bpl
 je .LBB2_10
.LBB_END2_7:
.LBB2_8:
 incq %r14
 movq %r14, %rdi
 movq %rbx, %rsi
 callq isMatch
 testl %eax, %eax
 setne %al
 jmp .LBB2_10
.LBB_END2_8:
.LBB2_9:
 xorl %eax, %eax
.LBB_END2_9:
.LBB2_10:
 popq %rbx
 .cfi_def_cfa_offset 24
 popq %r14
 .cfi_def_cfa_offset 16
 popq %rbp
 .cfi_def_cfa_offset 8
 .cfi_restore %rbx
 .cfi_restore %r14
 .cfi_restore %rbp
.LBB_END2_10:
.LBB2_11:
 movzbl %al, %eax
 retq
.LBB_END2_11:
.Lfunc_end2:
 .size isMatch, .Lfunc_end2-isMatch
 .cfi_endproc
 .section .llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text
 .byte 2                               # version
 .byte 0                               # feature
 .quad .Lfunc_begin2                   # function address
 .byte 12                              # number of basic blocks
 .byte 0                               # BB id
 .uleb128 .Lfunc_begin2-.Lfunc_begin2
 .uleb128 .LBB_END2_0-.Lfunc_begin2
 .byte 8
 .byte 1                               # BB id
 .uleb128 .LBB2_1-.LBB_END2_0
 .uleb128 .LBB_END2_1-.LBB2_1
 .byte 8
 .byte 2                               # BB id
 .uleb128 .LBB2_2-.LBB_END2_1
 .uleb128 .LBB_END2_2-.LBB2_2
 .byte 8
 .byte 5                               # BB id
 .uleb128 .LBB2_3-.LBB_END2_2
 .uleb128 .LBB_END2_3-.LBB2_3
 .byte 8
 .byte 10                              # BB id
 .uleb128 .LBB2_4-.LBB_END2_3
 .uleb128 .LBB_END2_4-.LBB2_4
 .byte 0
 .byte 3                               # BB id
 .uleb128 .LBB2_5-.LBB_END2_4
 .uleb128 .LBB_END2_5-.LBB2_5
 .byte 8
 .byte 4                               # BB id
 .uleb128 .LBB2_6-.LBB_END2_5
 .uleb128 .LBB_END2_6-.LBB2_6
 .byte 8
 .byte 8                               # BB id
 .uleb128 .LBB2_7-.LBB_END2_6
 .uleb128 .LBB_END2_7-.LBB2_7
 .byte 8
 .byte 6                               # BB id
 .uleb128 .LBB2_8-.LBB_END2_7
 .uleb128 .LBB_END2_8-.LBB2_8
 .byte 0
 .byte 9                               # BB id
 .uleb128 .LBB2_9-.LBB_END2_8
 .uleb128 .LBB_END2_9-.LBB2_9
 .byte 8
 .byte 13                              # BB id
 .uleb128 .LBB2_10-.LBB_END2_9
 .uleb128 .LBB_END2_10-.LBB2_10
 .byte 8
 .byte 7                               # BB id
 .uleb128 .LBB2_11-.LBB_END2_10
 .uleb128 .LBB_END2_11-.LBB2_11
 .byte 1
 .text
                                        # -- End function
 .globl bubbleSort                      # -- Begin function bubbleSort
 .p2align 4, 0x90
 .type bubbleSort,@function
bubbleSort:                             # @bubbleSort
.Lfunc_begin3:
 .cfi_startproc
# %bb.0:
                                        # kill: def $esi killed $esi def $rsi
 cmpl $1, %esi
 jne .LBB3_2
.LBB_END3_0:
.LBB3_1:
 retq
.LBB_END3_1:
.LBB3_2:
 leal -1(%rsi), %eax
 movq %rax, %rcx
 negq %rcx
 xorl %edx, %edx
 jmp .LBB3_4
.LBB_END3_2:
 .p2align 4, 0x90
.LBB3_3:                                #   in Loop: Header=BB3_4 Depth=1
 decl %esi
 decq %rax
 incq %rdx
 cmpl $1, %esi
 je .LBB3_1
.LBB_END3_3:
.LBB3_4:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_13 Depth 2
 cmpl $2, %esi
 jl .LBB3_3
.LBB_END3_4:
.LBB3_5:                                #   in Loop: Header=BB3_4 Depth=1
 movq %rdx, %r9
 notq %r9
 movl (%rdi), %r8d
 cmpq %rcx, %r9
 jne .LBB3_10
.LBB_END3_5:
.LBB3_6:                                #   in Loop: Header=BB3_4 Depth=1
 xorl %r9d, %r9d
.LBB_END3_6:
.LBB3_7:                                #   in Loop: Header=BB3_4 Depth=1
 testb $1, %al
 je .LBB3_3
.LBB_END3_7:
.LBB3_8:                                #   in Loop: Header=BB3_4 Depth=1
 movl 4(%rdi,%r9,4), %r10d
 cmpl %r10d, %r8d
 jle .LBB3_3
.LBB_END3_8:
.LBB3_9:                                #   in Loop: Header=BB3_4 Depth=1
 movl %r10d, (%rdi,%r9,4)
 movl %r8d, 4(%rdi,%r9,4)
 jmp .LBB3_3
.LBB_END3_9:
 .p2align 4, 0x90
.LBB3_10:                               #   in Loop: Header=BB3_4 Depth=1
 movq %rax, %r10
 andq $-2, %r10
 xorl %r9d, %r9d
 jmp .LBB3_13
.LBB_END3_10:
 .p2align 4, 0x90
.LBB3_11:                               #   in Loop: Header=BB3_13 Depth=2
 movl %r11d, 4(%rdi,%r9,4)
 movl %r8d, 8(%rdi,%r9,4)
.LBB_END3_11:
.LBB3_12:                               #   in Loop: Header=BB3_13 Depth=2
 addq $2, %r9
 cmpq %r9, %r10
 je .LBB3_7
.LBB_END3_12:
.LBB3_13:                               #   Parent Loop BB3_4 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
 movl 4(%rdi,%r9,4), %r11d
 cmpl %r11d, %r8d
 jle .LBB3_15
.LBB_END3_13:
.LBB3_14:                               #   in Loop: Header=BB3_13 Depth=2
 movl %r11d, (%rdi,%r9,4)
 movl %r8d, 4(%rdi,%r9,4)
 jmp .LBB3_16
.LBB_END3_14:
 .p2align 4, 0x90
.LBB3_15:                               #   in Loop: Header=BB3_13 Depth=2
 movl %r11d, %r8d
.LBB_END3_15:
.LBB3_16:                               #   in Loop: Header=BB3_13 Depth=2
 movl 8(%rdi,%r9,4), %r11d
 cmpl %r11d, %r8d
 jg .LBB3_11
.LBB_END3_16:
.LBB3_17:                               #   in Loop: Header=BB3_13 Depth=2
 movl %r11d, %r8d
 jmp .LBB3_12
.LBB_END3_17:
.Lfunc_end3:
 .size bubbleSort, .Lfunc_end3-bubbleSort
 .cfi_endproc
 .section .llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text
 .byte 2                               # version
 .byte 0                               # feature
 .quad .Lfunc_begin3                   # function address
 .byte 18                              # number of basic blocks
 .byte 0                               # BB id
 .uleb128 .Lfunc_begin3-.Lfunc_begin3
 .uleb128 .LBB_END3_0-.Lfunc_begin3
 .byte 8
 .byte 14                              # BB id
 .uleb128 .LBB3_1-.LBB_END3_0
 .uleb128 .LBB_END3_1-.LBB3_1
 .byte 1
 .byte 1                               # BB id
 .uleb128 .LBB3_2-.LBB_END3_1
 .uleb128 .LBB_END3_2-.LBB3_2
 .byte 0
 .byte 5                               # BB id
 .uleb128 .LBB3_3-.LBB_END3_2
 .uleb128 .LBB_END3_3-.LBB3_3
 .byte 8
 .byte 6                               # BB id
 .uleb128 .LBB3_4-.LBB_END3_3
 .uleb128 .LBB_END3_4-.LBB3_4
 .byte 8
 .byte 7                               # BB id
 .uleb128 .LBB3_5-.LBB_END3_4
 .uleb128 .LBB_END3_5-.LBB3_5
 .byte 8
 .byte 15                              # BB id
 .uleb128 .LBB3_6-.LBB_END3_5
 .uleb128 .LBB_END3_6-.LBB3_6
 .byte 8
 .byte 2                               # BB id
 .uleb128 .LBB3_7-.LBB_END3_6
 .uleb128 .LBB_END3_7-.LBB3_7
 .byte 8
 .byte 3                               # BB id
 .uleb128 .LBB3_8-.LBB_END3_7
 .uleb128 .LBB_END3_8-.LBB3_8
 .byte 8
 .byte 4                               # BB id
 .uleb128 .LBB3_9-.LBB_END3_8
 .uleb128 .LBB_END3_9-.LBB3_9
 .byte 0
 .byte 8                               # BB id
 .uleb128 .LBB3_10-.LBB_END3_9
 .uleb128 .LBB_END3_10-.LBB3_10
 .byte 0
 .byte 12                              # BB id
 .uleb128 .LBB3_11-.LBB_END3_10
 .uleb128 .LBB_END3_11-.LBB3_11
 .byte 8
 .byte 13                              # BB id
 .uleb128 .LBB3_12-.LBB_END3_11
 .uleb128 .LBB_END3_12-.LBB3_12
 .byte 8
 .byte 9                               # BB id
 .uleb128 .LBB3_13-.LBB_END3_12
 .uleb128 .LBB_END3_13-.LBB3_13
 .byte 8
 .byte 10                              # BB id
 .uleb128 .LBB3_14-.LBB_END3_13
 .uleb128 .LBB_END3_14-.LBB3_14
 .byte 0
 .byte 16                              # BB id
 .uleb128 .LBB3_15-.LBB_END3_14
 .uleb128 .LBB_END3_15-.LBB3_15
 .byte 8
 .byte 11                              # BB id
 .uleb128 .LBB3_16-.LBB_END3_15
 .uleb128 .LBB_END3_16-.LBB3_16
 .byte 8
 .byte 17                              # BB id
 .uleb128 .LBB3_17-.LBB_END3_16
 .uleb128 .LBB_END3_17-.LBB3_17
 .byte 0
 .text
                                        # -- End function
 .globl isPrime                         # -- Begin function isPrime
 .p2align 4, 0x90
 .type isPrime,@function
isPrime:                                # @isPrime
.Lfunc_begin4:
 .cfi_startproc
# %bb.0:
 xorl %eax, %eax
 cmpl $2, %edi
 jl .LBB4_9
.LBB_END4_0:
.LBB4_1:
 jne .LBB4_3
.LBB_END4_1:
.LBB4_2:
 movl $1, %eax
 retq
.LBB_END4_2:
.LBB4_3:
 testb $1, %dil
 je .LBB4_9
.LBB_END4_3:
.LBB4_4:
 movl $1, %eax
 cmpl $9, %edi
 jb .LBB4_9
.LBB_END4_4:
.LBB4_5:                                # %.preheader
 movl $5, %ecx
.LBB_END4_5:
 .p2align 4, 0x90
.LBB4_6:                                # =>This Inner Loop Header: Depth=1
 leal -2(%rcx), %esi
 movl %edi, %eax
 cltd
 idivl %esi
 testl %edx, %edx
 je .LBB4_8
.LBB_END4_6:
.LBB4_7:                                #   in Loop: Header=BB4_6 Depth=1
 movl %ecx, %eax
 imull %ecx, %eax
 addl $2, %ecx
 cmpl %edi, %eax
 jle .LBB4_6
.LBB_END4_7:
.LBB4_8:
 xorl %eax, %eax
 testl %edx, %edx
 setne %al
.LBB_END4_8:
.LBB4_9:
 retq
.LBB_END4_9:
.Lfunc_end4:
 .size isPrime, .Lfunc_end4-isPrime
 .cfi_endproc
 .section .llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text
 .byte 2                               # version
 .byte 0                               # feature
 .quad .Lfunc_begin4                   # function address
 .byte 10                              # number of basic blocks
 .byte 0                               # BB id
 .uleb128 .Lfunc_begin4-.Lfunc_begin4
 .uleb128 .LBB_END4_0-.Lfunc_begin4
 .byte 8
 .byte 1                               # BB id
 .uleb128 .LBB4_1-.LBB_END4_0
 .uleb128 .LBB_END4_1-.LBB4_1
 .byte 8
 .byte 10                              # BB id
 .uleb128 .LBB4_2-.LBB_END4_1
 .uleb128 .LBB_END4_2-.LBB4_2
 .byte 1
 .byte 2                               # BB id
 .uleb128 .LBB4_3-.LBB_END4_2
 .uleb128 .LBB_END4_3-.LBB4_3
 .byte 8
 .byte 3                               # BB id
 .uleb128 .LBB4_4-.LBB_END4_3
 .uleb128 .LBB_END4_4-.LBB4_4
 .byte 8
 .byte 4                               # BB id
 .uleb128 .LBB4_5-.LBB_END4_4
 .uleb128 .LBB_END4_5-.LBB4_5
 .byte 8
 .byte 5                               # BB id
 .uleb128 .LBB4_6-.LBB_END4_5
 .uleb128 .LBB_END4_6-.LBB4_6
 .byte 8
 .byte 8                               # BB id
 .uleb128 .LBB4_7-.LBB_END4_6
 .uleb128 .LBB_END4_7-.LBB4_7
 .byte 8
 .byte 6                               # BB id
 .uleb128 .LBB4_8-.LBB_END4_7
 .uleb128 .LBB_END4_8-.LBB4_8
 .byte 8
 .byte 7                               # BB id
 .uleb128 .LBB4_9-.LBB_END4_8
 .uleb128 .LBB_END4_9-.LBB4_9
 .byte 1
 .text
                                        # -- End function
 .globl main                            # -- Begin function main
 .p2align 4, 0x90
 .type main,@function
main:                                   # @main
.Lfunc_begin5:
 .cfi_startproc
# %bb.0:
 subq $280, %rsp                      # imm = 0x118
 .cfi_def_cfa_offset 288
 leaq 8(%rsp), %rax
 movw $111, 12(%rsp)
 movl $1819043176, 8(%rsp)            # imm = 0x6C6C6568
.LBB_END5_0:
 .p2align 4, 0x90
.LBB5_1:                                # =>This Inner Loop Header: Depth=1
 cmpb $0, 1(%rax)
 leaq 1(%rax), %rax
 jne .LBB5_1
.LBB_END5_1:
.LBB5_2:
 leaq -1(%rax), %rcx
 cmpq %rcx, %rax
 jae .LBB5_6
.LBB_END5_2:
.LBB5_3:
 movzbl -1(%rax), %ecx
 movb %cl, (%rax)
 movb $0, -1(%rax)
 leaq 1(%rax), %rdx
 leaq -2(%rax), %rcx
 cmpq %rcx, %rdx
 jae .LBB5_6
.LBB_END5_3:
.LBB5_4:                                # %.preheader
 addq $2, %rax
.LBB_END5_4:
 .p2align 4, 0x90
.LBB5_5:                                # =>This Inner Loop Header: Depth=1
 movzbl -1(%rax), %edx
 movzbl (%rcx), %esi
 movb %sil, -1(%rax)
 movb %dl, (%rcx)
 decq %rcx
 leaq 1(%rax), %rdx
 cmpq %rcx, %rax
 movq %rdx, %rax
 jb .LBB5_5
.LBB_END5_5:
.LBB5_6:
 movl $.L.str, %edi
 movl $.L.str.1, %esi
 callq isMatch
 movaps .L__const.main.rods(%rip), %xmm0
 movaps %xmm0, 48(%rsp)
 leaq 48(%rsp), %rdi
 movl $4, %esi
 callq tallestBillboard
 movaps .L__const.main.rods2(%rip), %xmm0
 movaps %xmm0, 16(%rsp)
 movabsq $25769803781, %rax              # imm = 0x600000005
 movq %rax, 32(%rsp)
 leaq 16(%rsp), %rdi
 movl $6, %esi
 callq tallestBillboard
 movaps .L__const.main.rods3+16(%rip), %xmm0
 movaps %xmm0, 176(%rsp)
 movaps .L__const.main.rods3(%rip), %xmm0
 movaps %xmm0, 160(%rsp)
 leaq 160(%rsp), %rdi
 movl $8, %esi
 callq tallestBillboard
 movaps .L__const.main.rods4+16(%rip), %xmm0
 movaps %xmm0, 80(%rsp)
 movaps .L__const.main.rods4(%rip), %xmm0
 movaps %xmm0, 64(%rsp)
 movl $9, 96(%rsp)
 leaq 64(%rsp), %rdi
 movl $9, %esi
 callq tallestBillboard
 movaps .L__const.main.rods5+16(%rip), %xmm0
 movaps %xmm0, 128(%rsp)
 movaps .L__const.main.rods5(%rip), %xmm0
 movaps %xmm0, 112(%rsp)
 movabsq $42949672969, %rax              # imm = 0xA00000009
 movq %rax, 144(%rsp)
 leaq 112(%rsp), %rdi
 movl $10, %esi
 callq tallestBillboard
 movaps .L__const.main.rods6+64(%rip), %xmm0
 movaps %xmm0, 256(%rsp)
 movaps .L__const.main.rods6+48(%rip), %xmm0
 movaps %xmm0, 240(%rsp)
 movaps .L__const.main.rods6+32(%rip), %xmm0
 movaps %xmm0, 224(%rsp)
 movaps .L__const.main.rods6+16(%rip), %xmm0
 movaps %xmm0, 208(%rsp)
 movaps .L__const.main.rods6(%rip), %xmm0
 movaps %xmm0, 192(%rsp)
 leaq 192(%rsp), %rdi
 movl $20, %esi
 callq tallestBillboard
 xorl %eax, %eax
 addq $280, %rsp                      # imm = 0x118
 .cfi_def_cfa_offset 8
 retq
.LBB_END5_6:
.Lfunc_end5:
 .size main, .Lfunc_end5-main
 .cfi_endproc
 .section .llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text
 .byte 2                               # version
 .byte 0                               # feature
 .quad .Lfunc_begin5                   # function address
 .byte 7                               # number of basic blocks
 .byte 0                               # BB id
 .uleb128 .Lfunc_begin5-.Lfunc_begin5
 .uleb128 .LBB_END5_0-.Lfunc_begin5
 .byte 8
 .byte 1                               # BB id
 .uleb128 .LBB5_1-.LBB_END5_0
 .uleb128 .LBB_END5_1-.LBB5_1
 .byte 8
 .byte 2                               # BB id
 .uleb128 .LBB5_2-.LBB_END5_1
 .uleb128 .LBB_END5_2-.LBB5_2
 .byte 8
 .byte 3                               # BB id
 .uleb128 .LBB5_3-.LBB_END5_2
 .uleb128 .LBB_END5_3-.LBB5_3
 .byte 8
 .byte 4                               # BB id
 .uleb128 .LBB5_4-.LBB_END5_3
 .uleb128 .LBB_END5_4-.LBB5_4
 .byte 8
 .byte 5                               # BB id
 .uleb128 .LBB5_5-.LBB_END5_4
 .uleb128 .LBB_END5_5-.LBB5_5
 .byte 8
 .byte 6                               # BB id
 .uleb128 .LBB5_6-.LBB_END5_5
 .uleb128 .LBB_END5_6-.LBB5_6
 .byte 1
 .text
                                        # -- End function
 .type .L__const.main.str,@object      # @__const.main.str
 .section .rodata.str1.1,"aMS",@progbits,1
.L__const.main.str:
 .asciz "hello"
 .size .L__const.main.str, 6

 .type .L.str,@object                  # @.str
.L.str:
 .asciz "aa"
 .size .L.str, 3

 .type .L.str.1,@object                # @.str.1
.L.str.1:
 .asciz ".*"
 .size .L.str.1, 3

 .type .L__const.main.rods,@object     # @__const.main.rods
 .section .rodata.cst16,"aM",@progbits,16
 .p2align 4, 0x0
.L__const.main.rods:
 .long 1                               # 0x1
 .long 2                               # 0x2
 .long 3                               # 0x3
 .long 6                               # 0x6
 .size .L__const.main.rods, 16

 .type .L__const.main.rods2,@object    # @__const.main.rods2
 .section .rodata,"a",@progbits
 .p2align 4, 0x0
.L__const.main.rods2:
 .long 1                               # 0x1
 .long 2                               # 0x2
 .long 3                               # 0x3
 .long 4                               # 0x4
 .long 5                               # 0x5
 .long 6                               # 0x6
 .size .L__const.main.rods2, 24

 .type .L__const.main.rods3,@object    # @__const.main.rods3
 .section .rodata.cst32,"aM",@progbits,32
 .p2align 4, 0x0
.L__const.main.rods3:
 .long 1                               # 0x1
 .long 2                               # 0x2
 .long 3                               # 0x3
 .long 4                               # 0x4
 .long 5                               # 0x5
 .long 6                               # 0x6
 .long 7                               # 0x7
 .long 8                               # 0x8
 .size .L__const.main.rods3, 32

 .type .L__const.main.rods4,@object    # @__const.main.rods4
 .section .rodata,"a",@progbits
 .p2align 4, 0x0
.L__const.main.rods4:
 .long 1                               # 0x1
 .long 2                               # 0x2
 .long 3                               # 0x3
 .long 4                               # 0x4
 .long 5                               # 0x5
 .long 6                               # 0x6
 .long 7                               # 0x7
 .long 8                               # 0x8
 .long 9                               # 0x9
 .size .L__const.main.rods4, 36

 .type .L__const.main.rods5,@object    # @__const.main.rods5
 .p2align 4, 0x0
.L__const.main.rods5:
 .long 1                               # 0x1
 .long 2                               # 0x2
 .long 3                               # 0x3
 .long 4                               # 0x4
 .long 5                               # 0x5
 .long 6                               # 0x6
 .long 7                               # 0x7
 .long 8                               # 0x8
 .long 9                               # 0x9
 .long 10                              # 0xa
 .size .L__const.main.rods5, 40

 .type .L__const.main.rods6,@object    # @__const.main.rods6
 .p2align 4, 0x0
.L__const.main.rods6:
 .long 1                               # 0x1
 .long 2                               # 0x2
 .long 3                               # 0x3
 .long 4                               # 0x4
 .long 5                               # 0x5
 .long 6                               # 0x6
 .long 7                               # 0x7
 .long 8                               # 0x8
 .long 9                               # 0x9
 .long 10                              # 0xa
 .long 11                              # 0xb
 .long 12                              # 0xc
 .long 13                              # 0xd
 .long 14                              # 0xe
 .long 15                              # 0xf
 .long 16                              # 0x10
 .long 17                              # 0x11
 .long 18                              # 0x12
 .long 19                              # 0x13
 .long 20                              # 0x14
 .size .L__const.main.rods6, 80

 .ident "Debian clang version 14.0.6"
 .section ".note.GNU-stack","",@progbits
