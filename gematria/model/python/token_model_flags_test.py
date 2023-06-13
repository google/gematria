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

from absl import flags
from absl.testing import flagsaver

import tensorflow.compat.v1 as tf

from gematria.model.python import token_model_flags

FLAGS = flags.FLAGS


class GetTokensFromCommandLineFlagsTest(tf.test.TestCase):

  @flagsaver.flagsaver
  def test_file_not_specified(self):
    # No tokens are returned when model_tokens is not provided.
    self.assertIsNone(token_model_flags.get_tokens_from_command_line_flags())
    # No tokens are returned even if model_tokens is provided.
    self.assertIsNone(
        token_model_flags.get_tokens_from_command_line_flags(
            ('FOO', 'BAR', 'BAZ')))

  @flagsaver.flagsaver
  def test_empty_file(self):
    model_tokens = ('MO', 'DEL')
    token_file = self.create_tempfile(content='')

    FLAGS.gematria_tokens_file = token_file.full_path

    loaded_tokens = token_model_flags.get_tokens_from_command_line_flags()
    self.assertEmpty(loaded_tokens)

    loaded_tokens = token_model_flags.get_tokens_from_command_line_flags(
        model_tokens)
    self.assertSequenceEqual(loaded_tokens, sorted(model_tokens))

  @flagsaver.flagsaver
  def test_file_with_tokens(self):
    file_tokens = ('FOO', 'BAR', 'BAZ')
    model_tokens = ('MO', 'DEL')
    # Separate tokens with an empty line, to test that empty lines are ignored.
    token_file = self.create_tempfile(content='\n\n'.join(file_tokens))

    FLAGS.gematria_tokens_file = token_file.full_path

    loaded_tokens = token_model_flags.get_tokens_from_command_line_flags()
    self.assertSequenceEqual(loaded_tokens, sorted(file_tokens))

    loaded_tokens = token_model_flags.get_tokens_from_command_line_flags(
        model_tokens)
    self.assertSequenceEqual(loaded_tokens, sorted(
        (*file_tokens, *model_tokens)))

  @flagsaver.flagsaver
  def test_file_with_comments(self):
    model_tokens = ('MO', 'DEL')
    file_contents = """
        # This is the first token.
        FOO

        # This is the second token.
        BAR

        # This is the third token.
        BAZ
        """
    token_file = self.create_tempfile(content=file_contents)

    FLAGS.gematria_tokens_file = token_file.full_path

    loaded_tokens = token_model_flags.get_tokens_from_command_line_flags()
    self.assertSequenceEqual(loaded_tokens, ('BAR', 'BAZ', 'FOO'))

    loaded_tokens = token_model_flags.get_tokens_from_command_line_flags(
        model_tokens)
    self.assertSequenceEqual(loaded_tokens, ('BAR', 'BAZ', 'DEL', 'FOO', 'MO'))


class GetOovTokenBehaviorFromCommandLineFlags(tf.test.TestCase):

  @flagsaver.flagsaver
  def test_default_and_empty(self):
    oov_token_behavior = (
        token_model_flags.get_oov_token_behavior_from_command_line_flags())
    self.assertEqual(oov_token_behavior.behavior_type,
                     oov_token_behavior.RETURN_ERROR)

    FLAGS.gematria_out_of_vocabulary_replacement_token = ''
    oov_token_behavior = (
        token_model_flags.get_oov_token_behavior_from_command_line_flags())
    self.assertEqual(oov_token_behavior.behavior_type,
                     oov_token_behavior.RETURN_ERROR)

  @flagsaver.flagsaver
  def test_some_token(self):
    token = 'FOO'
    FLAGS.gematria_out_of_vocabulary_replacement_token = token
    oov_token_behavior = (
        token_model_flags.get_oov_token_behavior_from_command_line_flags())
    self.assertEqual(oov_token_behavior.behavior_type,
                     oov_token_behavior.REPLACE_TOKEN)
    self.assertEqual(oov_token_behavior.replacement_token, token)


if __name__ == '__main__':
  tf.test.main()
