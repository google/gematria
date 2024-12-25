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
