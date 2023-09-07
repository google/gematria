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
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "gematria/datasets/block_wrapper.h"

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
// struct. Alignment / size of data types etc. isn't an issue here since this
// is only ever used for IPC with a forked process, so the ABI will be
// identical.
struct PipedData {
  uintptr_t code_address;
};

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
    return absl::InternalError("Read less than expected from pipe");
  }
  close(fd);
  return piped_data;
}

uintptr_t AlignDown(uintptr_t x, size_t align) { return x - (x % align); }

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

  return absl::InternalError(absl::StrFormat(
      "Child stopped with unexpected signal: %s", strsignal(signal)));
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

  accessed_addrs.code_location = pipe_data.value().code_address;

  return absl::OkStatus();
}

[[noreturn]] void ChildProcess(absl::Span<const uint8_t> basic_block,
                               int pipe_write_fd,
                               const AccessedAddrs& accessed_addrs) {
  // Make sure the parent is attached before doing anything that they might want
  // to listen for.
  ptrace(PTRACE_TRACEME, 0, nullptr, nullptr);
  raise(SIGSTOP);

  // Map all the locations we've previously discovered this code accesses.
  for (uintptr_t accessed_location : accessed_addrs.accessed_blocks) {
    auto location_ptr = reinterpret_cast<void*>(accessed_location);
    void* mapped_address =
        mmap(location_ptr, accessed_addrs.block_size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (mapped_address == MAP_FAILED) {
      perror("mapping previously discovered address failed");
      abort();
    }
    if (mapped_address != location_ptr) {
      fputs(
          "tried to map previously discovered address, but mmap couldn't map "
          "this address\n",
          stderr);
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

  PipedData piped_data = {.code_address =
                              reinterpret_cast<uintptr_t>(mapped_address)};
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

  auto mapped_func = reinterpret_cast<void (*)()>(mapped_address);
  mapped_func();

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

}  // namespace

// TODO(orodley):
// * Support blocks which access multiple addresses (need to re-execute with the
//   previously segfaulting address mapped until no segfaults).
// * Set up registers to minimise chance of needing to map an unmappable or
//   already mapped address, the communicate the necessary set of register in
//   order for the returned addresses to be accessed.
// * Be more robust against the code trying to access addresses that happen to
//   already be mapped upon forking the process, and therefore not segfaulting,
//   so we can't observe the access.
// * Determine when an address is relative to the instruction pointer, hence
//   may be different the next time it's executed if we load the code at a
//   different location (and/or return the address we loaded it at, which may be
//   necessary for the return addresses to be accessed).
// * Better error handling, return specific errors for different situations that
//   may occur, and document them well (e.g. handle SIGILL and return an error
//   stating that the code passed in is invalid, with a bad instruction at a
//   particular offset).
// * Much more complete testing.
absl::StatusOr<AccessedAddrs> FindAccessedAddrs(
    absl::Span<const uint8_t> basic_block) {
  AccessedAddrs accessed_addrs = {
      .code_location = 0,
      .block_size = static_cast<size_t>(getpagesize()),
      .accessed_blocks = {}};

  size_t num_accessed_blocks;
  do {
    num_accessed_blocks = accessed_addrs.accessed_blocks.size();
    auto status = ForkAndTestAddresses(basic_block, accessed_addrs);
    if (!status.ok()) {
      return status;
    }
  } while (accessed_addrs.accessed_blocks.size() != num_accessed_blocks);

  return accessed_addrs;
}

}  // namespace gematria
