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

from abc import ABC, abstractmethod
from collections.abc import Iterable
import os
import re
from enum import Enum


class BenchmarkScheduler(ABC):

  @abstractmethod
  def setup_and_get_benchmark_core(self) -> int | None:
    pass

  @abstractmethod
  def verify(self):
    pass


class NoSchedulingBenchmarkScheduler(BenchmarkScheduler):

  def setup_and_get_benchmark_core(self) -> int | None:
    return None

  def verify(self):
    pass


class DefaultBenchmarkScheduler(BenchmarkScheduler):

  def __init__(self):
    self._cpu_mask = []

  @staticmethod
  def _get_neighboring_threads(cpu_index: int) -> list[int]:
    with open(
        f'/sys/devices/system/cpu/cpu{cpu_index}/topology/thread_siblings_list'
    ) as thread_sibling_list_handle:
      neighboring_threads_strings = re.split(
          r'[-,]+', thread_sibling_list_handle.read().strip()
      )
      neighboring_threads = [
          int(cpu_index_str) for cpu_index_str in neighboring_threads_strings
      ]
    return neighboring_threads

  def _get_aux_core_and_hyperthread_pair(
      self,
      cpu_mask: Iterable[int],
  ) -> tuple[int, list[int]]:
    for cpu_index in cpu_mask:
      neighboring_threads = self._get_neighboring_threads(cpu_index)
      if len(neighboring_threads) != 2:
        raise ValueError('Expected two hyperthreads per CPU.')

      if (
          neighboring_threads[0] in cpu_mask
          and neighboring_threads[1] in cpu_mask
      ):
        cpus = list(cpu_mask)
        cpus.remove(neighboring_threads[0])
        cpus.remove(neighboring_threads[1])
        return (cpus[0], [neighboring_threads[0], neighboring_threads[1]])
    raise ValueError(
        'Expected a pair of neighboring hyperthreads in the CPU mask.'
    )

  def setup_and_get_benchmark_core(self) -> int | None:
    cpu_mask = os.sched_getaffinity(0)

    if len(cpu_mask) != 3:
      raise ValueError('Expected to have three CPUs.')

    aux_core, hyperthread_pair = self._get_aux_core_and_hyperthread_pair(
        cpu_mask
    )
    os.sched_setaffinity(0, [aux_core])
    self._cpu_mask = [aux_core]

    return hyperthread_pair[0]

  def verify(self):
    cpu_mask = list(os.sched_getaffinity(0))
    if self._cpu_mask != cpu_mask:
      raise ValueError('Expected the CPU mask to not change.')


class BenchmarkSchedulerImplementations(Enum):
  NoScheduling = 1
  Default = 2


def construct_benchmark_scheduler(
    scheduler_type: BenchmarkSchedulerImplementations,
) -> BenchmarkScheduler:
  if scheduler_type == BenchmarkSchedulerImplementations.NoScheduling:
    return NoSchedulingBenchmarkScheduler()
  elif scheduler_type == BenchmarkSchedulerImplementations.Default:
    return DefaultBenchmarkScheduler()
  else:
    raise ValueError('Unexpected Benchmark Scheduler Type.')
