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
from gematria.datasets.python import process_and_filter_bbs


class ProcessAndFilterBBsTests(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.bb_processor_filter = process_and_filter_bbs.BBProcessorFilter()

  def test_bb_processor(self):
    processed_bb = self.bb_processor_filter.remove_risky_instructions(
        "B801000000C3", "test", False
    )
    self.assertEqual(processed_bb, "B801000000")


if __name__ == "__main__":
  absltest.main()
