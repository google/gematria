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

#include "gematria/datasets/find_accessed_addrs.h"

#include <bits/types/siginfo_t.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "absl/random/random.h"
#include "absl/random/uniform_int_distribution.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gematria/datasets/block_wrapper.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/llvm/llvm_architecture_support.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegister.h"
#include "lib/Target/X86/MCTargetDesc/X86MCTargetDesc.h"

namespace gematria {
namespace {

// This is the address which we put the code at by default. This particular
// address is chosen because it's in the middle of a large empty range, under
// normal conditions, and we'd like rip-relative addressing to be likely to hit
// unmapped memory. If this address isn't available, we'll accept whatever the
// kernel gives us. But if possible, we use this address.
constexpr uintptr_t kDefaultCodeLocation = 0x2b00'0000'0000;

// The data which is communicated from the child to the parent. The protocol is
// that the child will either write nothing (if it crashes unexpectedly before
// getting the chance to write to the pipe), or it will write one copy of this
// struct. If the inner StatusCode is not OK, the rest of the fields other than
// status_message are undefined. Alignment / size of data types etc. isn't an
// issue here since this is only ever used for IPC with a forked process, so the
// ABI will be identical.
struct PipedData {
  absl::StatusCode status_code;
  char status_message[1024];
  uintptr_t code_address;
};

PipedData MakePipedData() {
  PipedData piped_data;

  // Zero out the entire object, not just each field individually -- we'll be
  // writing the entire thing out to the pipe as a byte array, and if we just
  // initialize all the fields we'll leave any padding uninitialized, which will
  // make msan unhappy when we write it to the pipe.
  memset(&piped_data, 0, sizeof(piped_data));
  return piped_data;
}

bool IsRetryable(int err) {
  return err == EINTR || err == EAGAIN || err == EWOULDBLOCK;
}

absl::Status WriteAll(int fd, const PipedData& piped_data) {
  auto data_span = absl::MakeConstSpan(
      reinterpret_cast<const uint8_t*>(&piped_data), sizeof piped_data);

  size_t current_offset = 0;
  while (current_offset < data_span.size()) {
    size_t to_write = data_span.size() - current_offset;

    ssize_t bytes_written;
    int err;
    do {
      bytes_written = write(fd, data_span.data() + current_offset, to_write);
      err = errno;
    } while (IsRetryable(err));

    if (bytes_written < 0) {
      return absl::ErrnoToStatus(err, "Failed to write to pipe");
    }

    current_offset += bytes_written;
  }

  close(fd);
  return absl::OkStatus();
}

absl::StatusOr<PipedData> ReadAll(int fd) {
  PipedData piped_data;
  auto data_span = absl::MakeSpan(reinterpret_cast<uint8_t*>(&piped_data),
                                  sizeof piped_data);

  size_t current_offset = 0;
  while (current_offset < data_span.size()) {
    size_t to_read = data_span.size() - current_offset;

    ssize_t bytes_read;
    int err;
    do {
      bytes_read = read(fd, data_span.data() + current_offset, to_read);
      err = errno;
    } while (IsRetryable(err));

    if (bytes_read < 0) {
      return absl::ErrnoToStatus(err, "Failed to read from pipe");
    }

    if (bytes_read == 0) {
      break;
    }
    current_offset += bytes_read;
  }

  if (current_offset != data_span.size()) {
    return absl::InternalError(absl::StrFormat(
        "Read less than expected from pipe (expected %uB, got %uB)",
        data_span.size(), current_offset));
  }
  close(fd);
  return piped_data;
}

uintptr_t AlignDown(uintptr_t x, size_t align) { return x - (x % align); }

void RandomiseRegs(absl::BitGen& gen, X64Regs& regs) {
  regs.ForEachReg([&gen](std::optional<int64_t>& value) {
    // Pick between three values: 0, a low address, and a high address. These
    // are picked to try to maximise the chance that some combination will
    // produce a valid address when run through a wide range of functions. This
    // is just a first stab, there are likely better sets of values we could use
    // here.
    constexpr int64_t kValues[] = {0, 0x15000, 0x1000000};
    absl::uniform_int_distribution<int> dist(0, std::size(kValues) - 1);
    value = kValues[dist(gen)];
  });
}

RawX64Regs ToRawRegs(const X64Regs& regs) {
  RawX64Regs raw_regs;
  raw_regs.rax = regs.rax.value_or(0);
  raw_regs.rbx = regs.rbx.value_or(0);
  raw_regs.rcx = regs.rcx.value_or(0);
  raw_regs.rdx = regs.rdx.value_or(0);
  raw_regs.rsi = regs.rsi.value_or(0);
  raw_regs.rdi = regs.rdi.value_or(0);
  raw_regs.rsp = regs.rsp.value_or(0);
  raw_regs.rbp = regs.rbp.value_or(0);
  raw_regs.r8 = regs.r8.value_or(0);
  raw_regs.r9 = regs.r9.value_or(0);
  raw_regs.r10 = regs.r10.value_or(0);
  raw_regs.r11 = regs.r11.value_or(0);
  raw_regs.r12 = regs.r12.value_or(0);
  raw_regs.r13 = regs.r13.value_or(0);
  raw_regs.r14 = regs.r14.value_or(0);
  raw_regs.r15 = regs.r15.value_or(0);

  return raw_regs;
}

std::string DumpRegs(const struct user_regs_struct& regs) {
  return absl::StrFormat(
      "\trsp=%016x rbp=%016x, rip=%016x\n"
      "\trax=%016x rbx=%016x, rcx=%016x\n"
      "\trdx=%016x rsi=%016x, rdi=%016x\n"
      "\t r8=%016x  r9=%016x, r10=%016x\n"
      "\tr11=%016x r12=%016x, r13=%016x\n"
      "\tr14=%016x r15=%016x",
      regs.rsp, regs.rbp, regs.rip, regs.rax, regs.rbx, regs.rcx, regs.rdx,
      regs.rsi, regs.rdi, regs.r8, regs.r9, regs.r10, regs.r11, regs.r12,
      regs.r13, regs.r14, regs.r15);
}

absl::Status ParentProcessInner(int child_pid, AccessedAddrs& accessed_addrs) {
  int status;
  waitpid(child_pid, &status, 0);

  if (!WIFSTOPPED(status)) {
    return absl::InternalError(absl::StrFormat(
        "Child terminated with an unexpected status: %d", status));
  }

  // At this point the child is stopped, and we are attached.
  // TODO(orodley): Since we don't set any ptrace options here, do we actually
  // need this initial stop and continue, or could the child just PTRACE_TRACEME
  // and keep going without raising an initial SIGSTOP?
  ptrace(PTRACE_CONT, child_pid, nullptr, nullptr);

  waitpid(child_pid, &status, 0);
  if (!WIFSTOPPED(status)) {
    return absl::InternalError(absl::StrFormat(
        "Child terminated with an unexpected status: %d", status));
  }

  int signal = WSTOPSIG(status);
  if (signal == SIGSEGV) {
    // SIGSEGV means the block tried to access some unmapped memory, as
    // expected.
    siginfo_t siginfo;
    ptrace(PTRACE_GETSIGINFO, child_pid, 0, &siginfo);
    uintptr_t addr = AlignDown(reinterpret_cast<uintptr_t>(siginfo.si_addr),
                               accessed_addrs.block_size);

    if (std::find(accessed_addrs.accessed_blocks.begin(),
                  accessed_addrs.accessed_blocks.end(),
                  addr) == accessed_addrs.accessed_blocks.end()) {
      accessed_addrs.accessed_blocks.push_back(addr);
    }
    return absl::OkStatus();
  }
  if (signal == SIGABRT) {
    // SIGABRT means the block finished, and executed our after-block code which
    // terminates the process. So, it didn't access any memory.
    return absl::OkStatus();
  }

  if (signal == SIGFPE) {
    // Floating point exceptions are potentially fixable by setting different
    // register values, so return 'Invalid argument', which communicates this.
    return absl::InvalidArgumentError("Floating point exception");
  }

  // Any other case is an unexpected signal, so let's capture the registers for
  // ease of debugging.
  struct user_regs_struct registers;
  ptrace(PTRACE_GETREGS, child_pid, 0, &registers);

  if (signal == SIGBUS) {
    siginfo_t siginfo;
    ptrace(PTRACE_GETSIGINFO, child_pid, 0, &siginfo);
    return absl::InternalError(absl::StrFormat(
        "Child stopped with unexpected signal: %s, address %ul\n%s",
        strsignal(signal), (uint64_t)siginfo.si_addr, DumpRegs(registers)));
  }
  return absl::InternalError(
      absl::StrFormat("Child stopped with unexpected signal: %s\n%s",
                      strsignal(signal), DumpRegs(registers)));
}

absl::Status ParentProcess(int child_pid, int pipe_read_fd,
                           AccessedAddrs& accessed_addrs) {
  auto result = ParentProcessInner(child_pid, accessed_addrs);

  // Regardless of what happened, kill the child with SIGKILL. If we just detach
  // with PTRACE_DETACH and let the process resume, it will exit with whatever
  // signal it was about to exit with before we caught it. If that signal is
  // SIGSEGV then it could get caught by (e.g.) the terminal and printed. We
  // don't want that as SIGSEGV is actually normal and expected here, and this
  // would just be useless noise.
  int err = kill(child_pid, SIGKILL);
  if (err != 0) {
    char* err_str = strerror(err);
    return absl::InternalError(
        absl::StrFormat("Failed to kill child process: %s", err_str));
  }
  // We must wait on the child after killing it, otherwise it remains as a
  // zombie process.
  waitpid(child_pid, nullptr, 0);

  if (!result.ok()) {
    return result;
  }

  auto pipe_data = ReadAll(pipe_read_fd);
  if (!pipe_data.ok()) {
    return pipe_data.status();
  }

  if (pipe_data->status_code != absl::StatusCode::kOk) {
    return absl::Status(pipe_data->status_code, pipe_data->status_message);
  }

  accessed_addrs.code_location = pipe_data.value().code_address;

  return absl::OkStatus();
}

// This is used over memcpy as memcpy may get unmapped. Doing the copy manually
// with a for loop doesn't help, as the compiler will often replace such loops
// with a call to memcpy.
void repmovsb(void* dst, const void* src, size_t count) {
  asm volatile("rep movsb" : "+D"(dst), "+S"(src), "+c"(count) : : "memory");
}

[[noreturn]] void AbortChildProcess(int pipe_write_fd, absl::Status status) {
  auto piped_data = MakePipedData();
  piped_data.status_code = status.code();

  // Write as much of the message as we can fit into the piped data struct. We
  // subtract one from the size to ensure we always leave a null-terminator on
  // the end.
  size_t message_length =
      std::min(status.message().length(), sizeof piped_data.status_message - 1);
  repmovsb(piped_data.status_message, status.message().data(), message_length);

  WriteAll(pipe_write_fd, piped_data).IgnoreError();
  abort();
}

// Keep this in sync with the code below that initialises mapped blocks. This
// value, when repeated over 8-byte chunks, should produce the same block as
// the code below.
constexpr uint64_t kBlockContents = 0x800000008;

[[noreturn]] void ChildProcess(absl::Span<const uint8_t> basic_block,
                               int pipe_write_fd,
                               const AccessedAddrs& accessed_addrs) {
  // Make sure the parent is attached before doing anything that they might want
  // to listen for.
  ptrace(PTRACE_TRACEME, 0, nullptr, nullptr);
  raise(SIGSTOP);

  // This value will turn up when reading from newly-mapped blocks (see below).
  // Unmap it so that we can correctly segfault and detect we've accessed it.
  // If it fails, oh well. Not worth aborting for as we might not even access
  // this address.
  munmap(reinterpret_cast<void*>(0x800000000), 0x10000);

  // Map all the locations we've previously discovered this code accesses.
  for (uintptr_t accessed_location : accessed_addrs.accessed_blocks) {
    auto location_ptr = reinterpret_cast<void*>(accessed_location);
    void* mapped_address =
        mmap(location_ptr, accessed_addrs.block_size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (mapped_address == MAP_FAILED) {
      AbortChildProcess(pipe_write_fd, absl::InternalError(absl::StrFormat(
                                           "mapping previously discovered "
                                           "address %p failed",
                                           location_ptr)));
    }
    if (mapped_address != location_ptr) {
      // Use InvalidArgument only for the case where we couldn't map an address.
      // This can happen when an address is computed based on registers and ends
      // up not being valid to map, which is potentially fixable by running
      // again with different register values. By using a unique error code we
      // can distinguish this case easily.
      AbortChildProcess(
          pipe_write_fd,
          absl::InvalidArgumentError(absl::StrFormat(
              "tried to map previously discovered address %p, but mmap "
              "couldn't map this address\n",
              (void*)location_ptr)));
    }

    // Initialise every fourth byte to 8, leaving the rest as zeroes. This
    // ensures that every aligned 8-byte chunk will contain 0x800000008, which
    // is a mappable address, and every 4-byte chunk will contain 0x8, which is
    // a non-zero value which won't give SIGFPE if used with div.
    uint8_t* block = reinterpret_cast<uint8_t*>(mapped_address);
    for (int i = 0; i < accessed_addrs.block_size; i += 4) {
      block[i] = 8;
    }
  }

  // We copy in our before-block code which sets up registers, followed by the
  // code we're given, followed by our after-block code which cleanly exits the
  // process. Otherwise if it finishes without segfaulting it will just run over
  // into whatever is afterwards.
  const auto before_block = GetGematriaBeforeBlockCode();
  const auto after_block = GetGematriaAfterBlockCode();
  const auto total_block_size =
      before_block.size() + basic_block.size() + after_block.size();

  uintptr_t desired_code_location = accessed_addrs.code_location;
  if (desired_code_location == 0) {
    desired_code_location = kDefaultCodeLocation;
  }

  void* mapped_address =
      mmap(reinterpret_cast<void*>(desired_code_location), total_block_size,
           PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapped_address == MAP_FAILED) {
    perror("mmap failed");
    abort();
  }

  auto piped_data = MakePipedData();
  piped_data.status_code = absl::OkStatus().code();
  piped_data.code_address = reinterpret_cast<uintptr_t>(mapped_address);
  auto status = WriteAll(pipe_write_fd, piped_data);
  if (!status.ok()) {
    abort();
  }

  absl::Span<uint8_t> mapped_span = absl::MakeSpan(
      reinterpret_cast<uint8_t*>(mapped_address), total_block_size);
  std::copy(before_block.begin(), before_block.end(), &mapped_span[0]);
  std::copy(basic_block.begin(), basic_block.end(),
            &mapped_span[before_block.size()]);
  std::copy(after_block.begin(), after_block.end(),
            &mapped_span[before_block.size() + basic_block.size()]);

  auto mapped_func = reinterpret_cast<void (*)(const RawX64Regs* initial_regs)>(
      mapped_address);
  auto raw_regs = ToRawRegs(accessed_addrs.initial_regs);
  mapped_func(&raw_regs);

  // mapped_func should never return, but we can't put [[noreturn]] on a
  // function pointer. So stick this here to satisfy the compiler.
  abort();
}

absl::Status ForkAndTestAddresses(absl::Span<const uint8_t> basic_block,
                                  AccessedAddrs& accessed_addrs) {
  int pipe_fds[2];
  if (pipe(pipe_fds) != 0) {
    int err = errno;
    return absl::ErrnoToStatus(
        err, "Failed to open pipe for communication with child process: %s");
  }
  int pipe_read_fd = pipe_fds[0];
  int pipe_write_fd = pipe_fds[1];

  pid_t pid = fork();
  switch (pid) {
    case -1: {
      int err = errno;
      return absl::ErrnoToStatus(err, "Failed to fork");
    }
    case 0:  // child
      // Child only writes to the pipe.
      close(pipe_read_fd);

      // ChildProcess doesn't return.
      ChildProcess(basic_block, pipe_write_fd, accessed_addrs);
    default:  // parent
      // Parent only reads from the pipe.
      close(pipe_write_fd);

      return ParentProcess(pid, pipe_read_fd, accessed_addrs);
  }
}

std::optional<int64_t>* LLVMRegNumberToX64Reg(X64Regs& regs,
                                              int llvm_reg_number) {
  switch (llvm_reg_number) {
    case llvm::X86::AH:
    case llvm::X86::AL:
    case llvm::X86::AX:
    case llvm::X86::HAX:
    case llvm::X86::EAX:
    case llvm::X86::RAX:
      return &regs.rax;

    case llvm::X86::BH:
    case llvm::X86::BL:
    case llvm::X86::BX:
    case llvm::X86::HBX:
    case llvm::X86::EBX:
    case llvm::X86::RBX:
      return &regs.rbx;

    case llvm::X86::CH:
    case llvm::X86::CL:
    case llvm::X86::CX:
    case llvm::X86::HCX:
    case llvm::X86::ECX:
    case llvm::X86::RCX:
      return &regs.rcx;

    case llvm::X86::DH:
    case llvm::X86::DL:
    case llvm::X86::DX:
    case llvm::X86::HDX:
    case llvm::X86::EDX:
    case llvm::X86::RDX:
      return &regs.rdx;

    case llvm::X86::SIH:
    case llvm::X86::SIL:
    case llvm::X86::SI:
    case llvm::X86::HSI:
    case llvm::X86::ESI:
    case llvm::X86::RSI:
      return &regs.rsi;

    case llvm::X86::DIH:
    case llvm::X86::DIL:
    case llvm::X86::DI:
    case llvm::X86::HDI:
    case llvm::X86::EDI:
    case llvm::X86::RDI:
      return &regs.rdi;

    case llvm::X86::SPH:
    case llvm::X86::SPL:
    case llvm::X86::SP:
    case llvm::X86::HSP:
    case llvm::X86::ESP:
    case llvm::X86::RSP:
      return &regs.rsp;

    case llvm::X86::BPH:
    case llvm::X86::BPL:
    case llvm::X86::BP:
    case llvm::X86::HBP:
    case llvm::X86::EBP:
    case llvm::X86::RBP:
      return &regs.rbp;

    case llvm::X86::R8B:
    case llvm::X86::R8BH:
    case llvm::X86::R8W:
    case llvm::X86::R8WH:
    case llvm::X86::R8D:
    case llvm::X86::R8:
      return &regs.r8;

    case llvm::X86::R9B:
    case llvm::X86::R9BH:
    case llvm::X86::R9W:
    case llvm::X86::R9WH:
    case llvm::X86::R9D:
    case llvm::X86::R9:
      return &regs.r9;

    case llvm::X86::R10B:
    case llvm::X86::R10BH:
    case llvm::X86::R10W:
    case llvm::X86::R10WH:
    case llvm::X86::R10D:
    case llvm::X86::R10:
      return &regs.r10;

    case llvm::X86::R11B:
    case llvm::X86::R11BH:
    case llvm::X86::R11W:
    case llvm::X86::R11WH:
    case llvm::X86::R11D:
    case llvm::X86::R11:
      return &regs.r11;

    case llvm::X86::R12B:
    case llvm::X86::R12BH:
    case llvm::X86::R12W:
    case llvm::X86::R12WH:
    case llvm::X86::R12D:
    case llvm::X86::R12:
      return &regs.r12;

    case llvm::X86::R13B:
    case llvm::X86::R13BH:
    case llvm::X86::R13W:
    case llvm::X86::R13WH:
    case llvm::X86::R13D:
    case llvm::X86::R13:
      return &regs.r13;

    case llvm::X86::R14B:
    case llvm::X86::R14BH:
    case llvm::X86::R14W:
    case llvm::X86::R14WH:
    case llvm::X86::R14D:
    case llvm::X86::R14:
      return &regs.r14;

    case llvm::X86::R15B:
    case llvm::X86::R15BH:
    case llvm::X86::R15W:
    case llvm::X86::R15WH:
    case llvm::X86::R15D:
    case llvm::X86::R15:
      return &regs.r15;

    // TODO(orodley): Implement these after adding support in X64Regs.

    // SSE registers, we don't support these yet but will later.
    case llvm::X86::XMM0:
    case llvm::X86::XMM1:
    case llvm::X86::XMM2:
    case llvm::X86::XMM3:
    case llvm::X86::XMM4:
    case llvm::X86::XMM5:
    case llvm::X86::XMM6:
    case llvm::X86::XMM7:
    case llvm::X86::XMM8:
    case llvm::X86::XMM9:
    case llvm::X86::XMM10:
    case llvm::X86::XMM11:
    case llvm::X86::XMM12:
    case llvm::X86::XMM13:
    case llvm::X86::XMM14:
    case llvm::X86::XMM15:

    // AVX registers, ditto.
    case llvm::X86::YMM0:
    case llvm::X86::YMM1:
    case llvm::X86::YMM2:
    case llvm::X86::YMM3:
    case llvm::X86::YMM4:
    case llvm::X86::YMM5:
    case llvm::X86::YMM6:
    case llvm::X86::YMM7:
    case llvm::X86::YMM8:
    case llvm::X86::YMM9:
    case llvm::X86::YMM10:
    case llvm::X86::YMM11:
    case llvm::X86::YMM12:
    case llvm::X86::YMM13:
    case llvm::X86::YMM14:
    case llvm::X86::YMM15:
    case llvm::X86::K0:
    case llvm::X86::K1:
    case llvm::X86::K2:
    case llvm::X86::K3:
    case llvm::X86::K4:
    case llvm::X86::K5:
    case llvm::X86::K6:
    case llvm::X86::K7:
    case llvm::X86::XMM16:
    case llvm::X86::XMM17:
    case llvm::X86::XMM18:
    case llvm::X86::XMM19:
    case llvm::X86::XMM20:
    case llvm::X86::XMM21:
    case llvm::X86::XMM22:
    case llvm::X86::XMM23:
    case llvm::X86::XMM24:
    case llvm::X86::XMM25:
    case llvm::X86::XMM26:
    case llvm::X86::XMM27:
    case llvm::X86::XMM28:
    case llvm::X86::XMM29:
    case llvm::X86::XMM30:
    case llvm::X86::XMM31:
    case llvm::X86::YMM16:
    case llvm::X86::YMM17:
    case llvm::X86::YMM18:
    case llvm::X86::YMM19:
    case llvm::X86::YMM20:
    case llvm::X86::YMM21:
    case llvm::X86::YMM22:
    case llvm::X86::YMM23:
    case llvm::X86::YMM24:
    case llvm::X86::YMM25:
    case llvm::X86::YMM26:
    case llvm::X86::YMM27:
    case llvm::X86::YMM28:
    case llvm::X86::YMM29:
    case llvm::X86::YMM30:
    case llvm::X86::YMM31:

    // AVX-512 registers, ditto.
    case llvm::X86::ZMM0:
    case llvm::X86::ZMM1:
    case llvm::X86::ZMM2:
    case llvm::X86::ZMM3:
    case llvm::X86::ZMM4:
    case llvm::X86::ZMM5:
    case llvm::X86::ZMM6:
    case llvm::X86::ZMM7:
    case llvm::X86::ZMM8:
    case llvm::X86::ZMM9:
    case llvm::X86::ZMM10:
    case llvm::X86::ZMM11:
    case llvm::X86::ZMM12:
    case llvm::X86::ZMM13:
    case llvm::X86::ZMM14:
    case llvm::X86::ZMM15:
    case llvm::X86::ZMM16:
    case llvm::X86::ZMM17:
    case llvm::X86::ZMM18:
    case llvm::X86::ZMM19:
    case llvm::X86::ZMM20:
    case llvm::X86::ZMM21:
    case llvm::X86::ZMM22:
    case llvm::X86::ZMM23:
    case llvm::X86::ZMM24:
    case llvm::X86::ZMM25:
    case llvm::X86::ZMM26:
    case llvm::X86::ZMM27:
    case llvm::X86::ZMM28:
    case llvm::X86::ZMM29:
    case llvm::X86::ZMM30:
    case llvm::X86::ZMM31:
    case llvm::X86::K0_K1:
    case llvm::X86::K2_K3:
    case llvm::X86::K4_K5:
    case llvm::X86::K6_K7:
    case llvm::X86::TMMCFG:
    case llvm::X86::TMM0:
    case llvm::X86::TMM1:
    case llvm::X86::TMM2:
    case llvm::X86::TMM3:
    case llvm::X86::TMM4:
    case llvm::X86::TMM5:
    case llvm::X86::TMM6:
    case llvm::X86::TMM7:

    // APX adds 16 more general purpose registers. Ditto.
    case llvm::X86::R16B:
    case llvm::X86::R16BH:
    case llvm::X86::R16W:
    case llvm::X86::R16WH:
    case llvm::X86::R16D:
    case llvm::X86::R16:
    case llvm::X86::R17B:
    case llvm::X86::R17BH:
    case llvm::X86::R17W:
    case llvm::X86::R17WH:
    case llvm::X86::R17D:
    case llvm::X86::R17:
    case llvm::X86::R18B:
    case llvm::X86::R18BH:
    case llvm::X86::R18W:
    case llvm::X86::R18WH:
    case llvm::X86::R18D:
    case llvm::X86::R18:
    case llvm::X86::R19B:
    case llvm::X86::R19BH:
    case llvm::X86::R19W:
    case llvm::X86::R19WH:
    case llvm::X86::R19D:
    case llvm::X86::R19:
    case llvm::X86::R20B:
    case llvm::X86::R20BH:
    case llvm::X86::R20W:
    case llvm::X86::R20WH:
    case llvm::X86::R20D:
    case llvm::X86::R20:
    case llvm::X86::R21B:
    case llvm::X86::R21BH:
    case llvm::X86::R21W:
    case llvm::X86::R21WH:
    case llvm::X86::R21D:
    case llvm::X86::R21:
    case llvm::X86::R22B:
    case llvm::X86::R22BH:
    case llvm::X86::R22W:
    case llvm::X86::R22WH:
    case llvm::X86::R22D:
    case llvm::X86::R22:
    case llvm::X86::R23B:
    case llvm::X86::R23BH:
    case llvm::X86::R23W:
    case llvm::X86::R23WH:
    case llvm::X86::R23D:
    case llvm::X86::R23:
    case llvm::X86::R24B:
    case llvm::X86::R24BH:
    case llvm::X86::R24W:
    case llvm::X86::R24WH:
    case llvm::X86::R24D:
    case llvm::X86::R24:
    case llvm::X86::R25B:
    case llvm::X86::R25BH:
    case llvm::X86::R25W:
    case llvm::X86::R25WH:
    case llvm::X86::R25D:
    case llvm::X86::R25:
    case llvm::X86::R26B:
    case llvm::X86::R26BH:
    case llvm::X86::R26W:
    case llvm::X86::R26WH:
    case llvm::X86::R26D:
    case llvm::X86::R26:
    case llvm::X86::R27B:
    case llvm::X86::R27BH:
    case llvm::X86::R27W:
    case llvm::X86::R27WH:
    case llvm::X86::R27D:
    case llvm::X86::R27:
    case llvm::X86::R28B:
    case llvm::X86::R28BH:
    case llvm::X86::R28W:
    case llvm::X86::R28WH:
    case llvm::X86::R28D:
    case llvm::X86::R28:
    case llvm::X86::R29B:
    case llvm::X86::R29BH:
    case llvm::X86::R29W:
    case llvm::X86::R29WH:
    case llvm::X86::R29D:
    case llvm::X86::R29:
    case llvm::X86::R30B:
    case llvm::X86::R30BH:
    case llvm::X86::R30W:
    case llvm::X86::R30WH:
    case llvm::X86::R30D:
    case llvm::X86::R30:
    case llvm::X86::R31B:
    case llvm::X86::R31BH:
    case llvm::X86::R31W:
    case llvm::X86::R31WH:
    case llvm::X86::R31D:
    case llvm::X86::R31:
      return nullptr;

    // The remainder of the LLVM x86 registers are not of interest to us, for
    // one reason or another, so we ignore them.
    default:
      return nullptr;
  }
}

}  // namespace

X64Regs FindReadRegs(const LlvmArchitectureSupport& llvm_arch_support,
                     absl::Span<const uint8_t> basic_block) {
  X64Regs regs;

  llvm::ArrayRef<uint8_t> llvm_array(basic_block.data(), basic_block.size());
  const auto instrs = DisassembleAllInstructions(
      llvm_arch_support.mc_disassembler(), llvm_arch_support.mc_instr_info(),
      llvm_arch_support.mc_register_info(),
      llvm_arch_support.mc_subtarget_info(),
      *llvm_arch_support.CreateMCInstPrinter(1), /*base_address=*/0,
      llvm_array);

  for (const auto& instr : *instrs) {
    const llvm::MCInst& mcinst = instr.mc_inst;
    const auto& mc_desc =
        llvm_arch_support.mc_instr_info().get(mcinst.getOpcode());

    for (int n = mc_desc.getNumDefs(); n < mcinst.getNumOperands(); n++) {
      const llvm::MCOperand& operand = mcinst.getOperand(n);
      if (!operand.isReg()) continue;

      auto reg = LLVMRegNumberToX64Reg(regs, operand.getReg());
      if (reg == nullptr) continue;

      *reg = 0;
    }

    for (const llvm::MCPhysReg llvm_reg : mc_desc.implicit_uses()) {
      auto reg = LLVMRegNumberToX64Reg(regs, llvm_reg);
      if (reg == nullptr) continue;

      *reg = 0;
    }
  }

  return regs;
}

// TODO(orodley):
// * Set up registers to minimise chance of needing to map an unmappable or
//   already mapped address, the communicate the necessary set of register in
//   order for the returned addresses to be accessed.
// * Be more robust against the code trying to access addresses that happen to
//   already be mapped upon forking the process, and therefore not segfaulting,
//   so we can't observe the access.
// * Better error handling, return specific errors for different situations that
//   may occur, and document them well (e.g. handle SIGILL and return an error
//   stating that the code passed in is invalid, with a bad instruction at a
//   particular offset).
// * Much more complete testing.
absl::StatusOr<AccessedAddrs> FindAccessedAddrs(
    absl::Span<const uint8_t> basic_block) {
    X64Regs initial_regs;
  initial_regs.ForEachReg([](std::optional<int64_t>& value) {
    // This value is chosen to be almost the lowest address that's able to be
    // mapped. We want it to be low so that even if a register is multiplied or
    // added to another register, it will still be likely to be within an
    // accessible region of memory. But it's very common to take small negative
    // offsets from a register as a memory address, so we want to leave some
    // space below so that such addresses will still be accessible.
    value = 0x15000;
  });

  absl::BitGen gen;

  AccessedAddrs accessed_addrs = {
      .code_location = 0,
      .block_size = static_cast<size_t>(getpagesize()),
      .block_contents = kBlockContents,
      .accessed_blocks = {},
      .initial_regs = initial_regs,
  };

  int n = 0;
  size_t num_accessed_blocks;
  do {
    num_accessed_blocks = accessed_addrs.accessed_blocks.size();
    auto status = ForkAndTestAddresses(basic_block, accessed_addrs);
    if (absl::IsInvalidArgument(status)) {
      if (n > 100) {
        return status;
      }

      accessed_addrs.accessed_blocks.clear();
      RandomiseRegs(gen, accessed_addrs.initial_regs);
    } else if (!status.ok()) {
      return status;
    }

    n++;
  } while (accessed_addrs.accessed_blocks.size() != num_accessed_blocks);

  return accessed_addrs;
}

}  // namespace gematria
