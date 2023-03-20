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

"""Tests for the Clif wrapper of OutOfVocabularyTokenBehavior."""

from absl.testing import absltest

from gematria.model.python import oov_token_behavior

_OutOfVocabularyTokenBehavior = oov_token_behavior.OutOfVocabularyTokenBehavior


class OutOfVocabularyTokenBehaviorTest(absltest.TestCase):

  def test_return_error(self):
    return_error = _OutOfVocabularyTokenBehavior.return_error()
    self.assertIsNotNone(return_error)
    self.assertEqual(
        return_error.behavior_type,
        _OutOfVocabularyTokenBehavior.BehaviorType.RETURN_ERROR,
    )
    self.assertEmpty(return_error.replacement_token)

  def test_replace_with_token(self):
    replacement_token = 'FooBar'
    replace_with_token = _OutOfVocabularyTokenBehavior.replace_with_token(
        replacement_token
    )
    self.assertIsNotNone(replace_with_token)
    self.assertEqual(
        replace_with_token.behavior_type,
        _OutOfVocabularyTokenBehavior.BehaviorType.REPLACE_TOKEN,
    )
    self.assertEqual(replace_with_token.replacement_token, replacement_token)


if __name__ == '__main__':
  absltest.main()
