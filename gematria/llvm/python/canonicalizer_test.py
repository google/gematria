# Copyright 2023 Google Inc.
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

from gematria.llvm.python import canonicalizer
from gematria.llvm.python import llvm_architecture_support


class CanonicalizerTest(absltest.TestCase):

  def test_x86_64(self):
    llvm = llvm_architecture_support.LlvmArchitectureSupport.x86_64()
    x86_canonicalizer = canonicalizer.Canonicalizer.x86_64(llvm)
    self.assertIsInstance(x86_canonicalizer, canonicalizer.Canonicalizer)


if __name__ == "__main__":
  absltest.main()
