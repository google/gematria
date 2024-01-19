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

.global gematria_before_block
.type gematria_before_block, %function
// This code is copied directly before the code we execute, as we need exact
// control over the code that is executed, so we can't do it from C++. Its job
// is to initialise the registers to consistent values so that the accessed
// addresses don't depend on the register state when calling FindAccessedAddrs.
//
// It takes one argument, in rdi: a pointer to a struct containing the initial
// register values for this execution.
// TODO(orodley): Update to support r16-r31.
gematria_before_block:
  mov r15, rdi
  mov rax, [r15 + 0x00]
  mov rbx, [r15 + 0x08]
  mov rcx, [r15 + 0x10]
  mov rdx, [r15 + 0x18]
  mov rsi, [r15 + 0x20]
  mov rdi, [r15 + 0x28]
  mov rsp, [r15 + 0x30]
  mov rbp, [r15 + 0x38]
  mov r8,  [r15 + 0x40]
  mov r9,  [r15 + 0x48]
  mov r10, [r15 + 0x50]
  mov r11, [r15 + 0x58]
  mov r12, [r15 + 0x60]
  mov r13, [r15 + 0x68]
  mov r14, [r15 + 0x70]
  mov r15, [r15 + 0x78]

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
