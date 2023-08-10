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

size = . - gematria_after_block
.size gematria_after_block, size

// Store the size of gematria_after_block, so we know how much to copy.
.rodata
.global gematria_after_block_size
gematria_after_block_size:
  .quad size
