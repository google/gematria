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

from absl.testing import absltest

from gematria.datasets.python import bhive_to_exegesis
from gematria.datasets.python import exegesis_benchmark
from gematria.llvm.python import llvm_architecture_support


class ExegesisBenchmarkTests(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self._x86_llvm = llvm_architecture_support.LlvmArchitectureSupport.x86_64()
    self.bhive_to_exegesis = bhive_to_exegesis.BHiveToExegesis.create(
        self._x86_llvm
    )
    self.exegesis_benchmark = exegesis_benchmark.ExegesisBenchmark.create()

  def test_benchmarking(self):
    block_annotations = self.bhive_to_exegesis.annotate_basic_block(
        "4829d38b44246c8b54246848c1fb034829d04839c3",
        bhive_to_exegesis.AnnotatorType.fast,
        50,
    )

    benchmark_code = self.exegesis_benchmark.process_annotated_block(
        "4829d38b44246c8b54246848c1fb034829d04839c3", block_annotations
    )

    block_measurement = self.exegesis_benchmark.benchmark_basic_block(
        benchmark_code
    )

    self.assertLess(block_measurement, 10)


if __name__ == "__main__":
  absltest.main()
