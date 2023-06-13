# Copyright 2022 Google Inc.
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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from gematria.testing.python import matchers

# The tests use explicit comparisons to test their behavior.
# pylint: disable=g-generic-assert


class MatchersTest(parameterized.TestCase):

  @parameterized.parameters(
      (matchers.SequenceEqual(()), ()),
      (matchers.SequenceEqual(()), []),
      (matchers.SequenceEqual([]), ()),
      (matchers.SequenceEqual([]), []),
      (matchers.SequenceEqual((1, 'hello')), (1, 'hello')),
      (matchers.SequenceEqual((1, 'hello')), [1, 'hello']),
      (matchers.SequenceEqual([mock.ANY, 'hello']), (1, 'hello')),
  )
  def testSequenceEqualMatches(self, matcher, value):
    self.assertTrue(matcher == value)
    self.assertFalse(matcher != value)

  @parameterized.parameters(
      (matchers.SequenceEqual(()), (1, 2, 3)),
      (matchers.SequenceEqual(()), None),
      (matchers.SequenceEqual([1]), 1),
      (matchers.SequenceEqual(('a', 'b', 'c')), ('a', 'b', 3)),
  )
  def testSequenceEqualDoesNotMatch(self, matcher, value):
    self.assertTrue(matcher != value)
    self.assertFalse(matcher == value)


if __name__ == '__main__':
  absltest.main()
