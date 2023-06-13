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
import itertools

from absl.testing import absltest
from gematria.utils.python import flag_utils


class FlagUtilsTest(absltest.TestCase):

  def test_layers_from_str(self):
    values = ("", "256", "128,256", "32, 33, 34")
    expected_layers = ((), (256,), (128, 256), (32, 33, 34))
    for value, expected in itertools.zip_longest(values, expected_layers):
      self.assertSequenceEqual(expected, flag_utils.layers_from_str(value))

  def test_is_between_zero_and_one(self):
    values = (-1.0, 0.0, -0.0, 0.1, 0.999999, 1.0, 1.000001, 100.0)
    expected_outputs = (False, True, True, True, True, True, False, False)
    for value, expected in itertools.zip_longest(values, expected_outputs):
      self.assertEqual(expected, flag_utils.is_between_zero_and_one(value))

  def test_is_positive(self):
    values = (-1.0, 0.0, -0.0, 0.1, 1.0, 10.0)
    expected_outputs = (False, False, False, True, True, True)
    for value, expected in itertools.zip_longest(values, expected_outputs):
      self.assertEqual(expected, flag_utils.is_positive(value))

  def test_is_positive_integer_list(self):
    values = (
        "255",
        "128,255",
        "1, 2, 3",
        "",
        "-1, 10",
        "0.1, 0.2",
        "256,foo,128",
    )
    expected_outputs = (True, True, True, True, False, False, False)
    for value, expected in itertools.zip_longest(values, expected_outputs):
      self.assertEqual(expected, flag_utils.is_positive_integer_list(value))


if __name__ == "__main__":
  absltest.main()
