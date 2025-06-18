// Copyright 2023 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

.intel_syntax noprefix

.text

.global gematria_prologue
.type gematria_prologue, %function
// This code is copied directly before the code we execute, as we need exact
// control over the code that is executed, so we can't do it from C++. Its job
// is to initialise the registers to consistent values so that the accessed
// addresses don't depend on the register state when calling FindAccessedAddrs.
//
// See the "WrappedFunc" typedef for the function signature this code has. Since
// it doesn't return we make no guarantees about preserving registers / stack
// frame, but we do use the normal calling convention for input parameters.
gematria_prologue:
  mov rax, [rdi] // rax = vector_reg_width
  mov rbx, [rdi + 8] // rbx = uses_upper_vector_regs
  mov rcx, [rdi + 16] // rcx = uses_apx_regs
  cmp rax, 0
  je set_int_registers
  cmp rax, 1
  je set_xmm_registers
  cmp rax, 2
  je set_ymm_registers
  jmp set_zmm_registers

set_xmm_registers:
.irp n, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
  vpbroadcastq xmm\n, [rdi + 0x118 + (8 * \n)]
.endr
  cmp rbx, 0
  je set_int_registers
set_upper_xmm_registers:
.irp n, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
  vpbroadcastq xmm\n, [rdi + 0x118 + (8 * \n)]
.endr
  jmp set_int_registers

set_ymm_registers:
.irp n, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
  vpbroadcastq ymm\n, [rdi + 0x118 + (8 * \n)]
.endr
  cmp rbx, 0
  je set_int_registers
.irp n, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
  vpbroadcastq ymm\n, [rdi + 0x118 + (8 * \n)]
.endr
  jmp set_int_registers

set_zmm_registers:
.irp n, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
  vpbroadcastq zmm\n, [rdi + 0x118 + (8 * \n)]
.endr
  cmp rbx, 0
  je set_int_registers
.irp n, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
  vpbroadcastq zmm\n, [rdi + 0x118 + (8 * \n)]
.endr

set_int_registers:
  mov r15, rdi
  // Set APX registers first if necessary, as we need to overwrite RDI as part
  // of setting the base registers.
  cmp rcx, 0
  je set_base_registers
.irp n, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
  mov r\n, [rdi + 0x98 + (8 * \n)]
.endr

set_base_registers:
  mov rax, [r15 + 0x18]
  mov rbx, [r15 + 0x20]
  mov rcx, [r15 + 0x28]
  mov rdx, [r15 + 0x30]
  mov rsi, [r15 + 0x38]
  mov rdi, [r15 + 0x40]
  mov rsp, [r15 + 0x48]
  mov rbp, [r15 + 0x50]
  mov r8,  [r15 + 0x58]
  mov r9,  [r15 + 0x60]
  mov r10, [r15 + 0x68]
  mov r11, [r15 + 0x70]
  mov r12, [r15 + 0x78]
  mov r13, [r15 + 0x80]
  mov r14, [r15 + 0x88]
  mov r15, [r15 + 0x90]

_gematria_prologue_size = . - gematria_prologue
.size gematria_prologue, _gematria_prologue_size

// This code is copied directly after the code we execute, as we no longer have
// a stack to hold a return address. Its job is just to cleanly exit the process
// in a way that is clearly distinguishable, so we can tell that the block
// finished without any segfaults, illegal instructions, or other errors.
//
// We do this by raising SIGABRT. We can't call any standard library functions,
// as we don't have a stack. So we have to issue the syscalls manually.
.global gematria_epilogue
.type gematria_epilogue, %function
gematria_epilogue:
  // getpid()
  mov rax, 39
  syscall

  // kill(getpid(), SIGABRT)
  mov rdi, rax
  mov rsi, 6
  mov rax, 62
  syscall

  // The above syscall should terminate the process and never return. Just in
  // case it does return, make sure we crash and don't start executing whatever
  // random bytes are next.
  ud2

_gematria_epilogue_size = . - gematria_epilogue
.size gematria_epilogue, _gematria_epilogue_size

.rodata
// Store the size of gematria_prologue, so we know how much to copy.
.global gematria_prologue_size
gematria_prologue_size:
  .quad _gematria_prologue_size

// Ditto for gematria_after_block.
.global gematria_epilogue_size
gematria_epilogue_size:
  .quad _gematria_epilogue_size
