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

// This value is chosen to be one of the lowest addresses that's able to be
// mapped. We want it to be low so that even if a register is multiplied or
// added to another register, it will still be likely to be within an
// accessible region of memory.
INITIAL_REG_VALUE = 0x10000

.global gematria_before_block
.type gematria_before_block, %function
// This code is copied directly after the code we execute, as we no longer have
// a stack to hold a return address. Its job is initialise the registers to
// consistent values so that the accessed addresses don't depend on the register
// state when calling FindAccessedAddrs.
gematria_before_block:
  mov rax, INITIAL_REG_VALUE
  mov rbx, INITIAL_REG_VALUE
  mov rcx, INITIAL_REG_VALUE
  mov rdx, INITIAL_REG_VALUE
  mov rsi, INITIAL_REG_VALUE
  mov rdi, INITIAL_REG_VALUE
  mov rsp, INITIAL_REG_VALUE
  mov rbp, INITIAL_REG_VALUE
  mov r8, INITIAL_REG_VALUE
  mov r9, INITIAL_REG_VALUE
  mov r10, INITIAL_REG_VALUE
  mov r11, INITIAL_REG_VALUE
  mov r12, INITIAL_REG_VALUE
  mov r13, INITIAL_REG_VALUE
  mov r14, INITIAL_REG_VALUE
  mov r15, INITIAL_REG_VALUE

_gematria_before_block_size = . - gematria_before_block
.size gematria_before_block, _gematria_before_block_size

// This code is copied directly after the code we execute, as we no longer have
// a stack to hold a return address. Its job is just to cleanly exit the process
// in a way that is clearly distinguishable, so we can tell that the block
// finished without any segfaults, illegal instructions, or other errors.
//
// We do this by raising SIGABRT. We can't call any standard library functions,
// as we don't have a stack. So we have to issue the syscalls manually.
.global gematria_after_block
.type gematria_after_block, %function
gematria_after_block:
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

_gematria_after_block_size = . - gematria_after_block
.size gematria_after_block, _gematria_after_block_size

.rodata
// Store the size of gematria_before_block, so we know how much to copy.
.global gematria_after_block_size
gematria_after_block_size:
  .quad _gematria_after_block_size

// Ditto for gematria_after_block.
.global gematria_before_block_size
gematria_before_block_size:
  .quad _gematria_before_block_size
