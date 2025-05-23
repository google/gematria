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

import os
from collections.abc import Iterable

from absl.testing import absltest

from gematria.datasets.pipelines import benchmark_cpu_scheduler


class BenchmarkSchedulerTests(absltest.TestCase):

  def _set_normal_affinity(self):
    cpu_mask = os.sched_getaffinity(0)
    cpu_mask_list = list(cpu_mask)

    original_cpu_mask = list(cpu_mask_list)
    self.addCleanup(lambda: os.sched_setaffinity(0, original_cpu_mask))

    # We first select a CPU to use for the hyperthread pair and then remove
    # both cores in the pair from the mask set to ensure that we do not
    # accidentally select an auxiliary CPU that is part of the pair.
    hyperthread_pair_part = next(iter(cpu_mask))
    hyperthread_pair = benchmark_cpu_scheduler._get_neighboring_threads(
        hyperthread_pair_part
    )
    cpu_mask.remove(hyperthread_pair[0])
    cpu_mask.remove(hyperthread_pair[1])
    aux_cpu = cpu_mask.pop()
    new_cpu_mask = [aux_cpu, *hyperthread_pair]

    os.sched_setaffinity(0, new_cpu_mask)
    return (aux_cpu, hyperthread_pair, cpu_mask_list)

  def test_no_scheduling(self):
    scheduler = benchmark_cpu_scheduler.NoSchedulingBenchmarkScheduler()
    self.assertIsNone(scheduler.setup_and_get_benchmark_core())
    scheduler.verify()

  def test_default_scheduler_get_neighboring_threads(self):
    neighboring_threads = benchmark_cpu_scheduler._get_neighboring_threads(0)

    # Just check that we get two CPU ids back that are not the same. We cannot
    # do much more without knowing more about the system topology, and this
    # should be a reasonable enough test.
    self.assertLen(neighboring_threads, 2)
    self.assertNotEqual(neighboring_threads[0], neighboring_threads[1])

  def test_default_scheduler_get_cores(self):
    expected_aux_cpu, expected_hyperthread_pair, _ = self._set_normal_affinity()
    scheduler = benchmark_cpu_scheduler.DefaultBenchmarkScheduler()
    cpu_mask = os.sched_getaffinity(0)
    aux_cpu, hyperthread_pair = scheduler._get_aux_core_and_hyperthread_pair(
        cpu_mask
    )
    self.assertEqual(aux_cpu, expected_aux_cpu)
    self.assertContainsSubsequence(hyperthread_pair, expected_hyperthread_pair)

  def test_default_scheduler_get_cores_no_neighboring_threads(self):
    cpu_mask = os.sched_getaffinity(0)

    # If we have less than five CPUs, that means we are guaranteed to get
    # a pair when selecting three of them. Skip this test in those
    # instances.
    if len(cpu_mask) < 5:
      self.skipTest('Not enough cores to complete setup properly.')

    three_cores = [cpu_mask.pop(), cpu_mask.pop(), cpu_mask.pop()]

    scheduler = benchmark_cpu_scheduler.DefaultBenchmarkScheduler()
    with self.assertRaisesRegex(
        ValueError,
        'Expected a pair of neighboring hyperthreads and an aux thread in the'
        ' CPU mask.',
    ):
      scheduler._get_aux_core_and_hyperthread_pair(three_cores)

  def test_default_scheduler_setup(self):
    expected_aux_cpu, expected_hyperthread_pair, _ = self._set_normal_affinity()

    scheduler = benchmark_cpu_scheduler.DefaultBenchmarkScheduler()
    benchmark_core = scheduler.setup_and_get_benchmark_core()
    self.assertIn(benchmark_core, expected_hyperthread_pair)
    set_cpu_mask = os.sched_getaffinity(0)
    self.assertSequenceEqual(
        set_cpu_mask,
        {
            expected_aux_cpu,
        },
    )

  def test_default_scheduler_not_three_cpus(self):
    old_cpu_mask = os.sched_getaffinity(0)
    cpu_mask_list = list(old_cpu_mask)
    os.sched_setaffinity(0, cpu_mask_list[0:2])

    scheduler = benchmark_cpu_scheduler.DefaultBenchmarkScheduler()
    with self.assertRaises(ValueError):
      scheduler.setup_and_get_benchmark_core()

    os.sched_setaffinity(0, old_cpu_mask)

  def test_default_scheduler_verify(self):
    self._set_normal_affinity()

    scheduler = benchmark_cpu_scheduler.DefaultBenchmarkScheduler()
    scheduler.setup_and_get_benchmark_core()
    scheduler.verify()

  def test_default_scheduler_verify_mask_changed(self):
    _, _, old_cpu_mask = self._set_normal_affinity()

    scheduler = benchmark_cpu_scheduler.DefaultBenchmarkScheduler()
    scheduler.setup_and_get_benchmark_core()

    cpu_mask_list = list(old_cpu_mask)
    os.sched_setaffinity(0, cpu_mask_list[1:3])
    with self.assertRaises(ValueError):
      scheduler.verify()


if __name__ == '__main__':
  absltest.main()
