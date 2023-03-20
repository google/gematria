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

from gematria.utils.python import timer


class TimerTest(absltest.TestCase):

  @mock.patch('absl.logging.info')
  @mock.patch('time.time', side_effect=[10, 15])
  def test_timer_one_iteration_absl_logging(self, mock_time, mock_logging_info):
    del mock_time  # Unused
    timer_name = 'My timer'
    with timer.scoped(timer_name):
      pass

    self.assertEqual(mock_logging_info.call_count, 1)
    log_args, _ = mock_logging_info.call_args
    self.assertEqual(log_args, ('%s: %fs', timer_name, 5))

  @mock.patch('absl.logging.info')
  @mock.patch('time.time', side_effect=[10, 15])
  def test_timer_one_iteration_custom_log_function(
      self, mock_time, mock_logging_info
  ):
    del mock_time  # Unused
    timer_name = 'My other timer'
    mock_log_function = mock.MagicMock()
    with timer.scoped(timer_name, log_function=mock_log_function):
      pass

    self.assertFalse(mock_logging_info.called)

    self.assertEqual(mock_log_function.call_count, 1)
    log_args, _ = mock_log_function.call_args
    self.assertEqual(log_args, ('%s: %fs', timer_name, 5))

  @mock.patch('absl.logging.info')
  @mock.patch('time.time', side_effect=[12, 18])
  def test_timer_many_iterations(self, mock_time, mock_logging_info):
    del mock_time  # Unused
    timer_name = 'My other timer'
    with timer.scoped(timer_name, num_iterations=3):
      pass

    self.assertEqual(mock_logging_info.call_count, 1)
    log_args, _ = mock_logging_info.call_args
    self.assertEqual(log_args, ('%s: %fs, %fs per iteration', timer_name, 6, 2))


if __name__ == '__main__':
  absltest.main()
