# Copyright 2024 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing_extensions import override


class BenchmarkScheduler(metaclass=abc.ABCMeta):
  """Schedules a benchmark and the parent process to reduce noise.

  BenchmarkScheduler is an abstraction that provides two main pieces of
  functionality. Firstly, it provides a function
  (setup_and_get_benchmark_core) that allows an implementation to perform any
  necessary setup in the parent process and provide a core ID that should be
  used to perform any benchmarking. Additionally, implementations are
  intended to hold state to verify that the expected state is maintained and
  not changed by external software.
  """

  @abc.abstractmethod
  def setup_and_get_benchmark_core(self) -> int | None:
    """Sets up the parent process and chooses a benchmark core.

    This function will perform any relevant setup in the parent process,
    and return a core ID that can be used to run benchmarks on.

    Returns:
      Returns an integer core ID to specify a core that should be used for
      running benchmarks, or None to indicate that any core can be used.
    """

  @abc.abstractmethod
  def verify(self):
    """Verifies that conditions match what is expected.

    This function allows for implementations to verify that the original
    setup created in setup_and_get_benchmark_core is maintained for every
    benchmark that is run.
    """


class NoSchedulingBenchmarkScheduler(BenchmarkScheduler):
  """A basic BenchmarkScheduler implementation that does nothing.

  This BenchmarkScheduler implementation does nothing. It leaves scheduling
  of the benchmarking process to the operating system by specifying any core
  can be used for benchmarking, performs no setup in the parent process, and
  performs no verification.
  """

  @override
  def setup_and_get_benchmark_core(self) -> int | None:
    return None

  @override
  def verify(self):
    pass
