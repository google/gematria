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
"""Sanity checks for the test data."""

from absl.testing import absltest

from gematria.proto import throughput_pb2
from gematria.testing.python import basic_blocks_with_throughput


def _cleanup(basic_block: throughput_pb2.BasicBlockWithThroughputProto):
  del basic_block.inverse_throughputs[:]
  return basic_block


class BasicBlocksWithThroughputTest(absltest.TestCase):

  def test_data_is_present(self):
    """Checks that the returned list is not empty."""
    blocks = basic_blocks_with_throughput.get_basic_blocks()
    self.assertNotEmpty(blocks)

  def test_data_is_isolated(self):
    """Checks that two returned copies of the data are independent."""
    blocks1 = basic_blocks_with_throughput.get_basic_blocks()
    self.assertGreaterEqual(len(blocks1), 2)
    blocks2 = basic_blocks_with_throughput.get_basic_blocks()
    self.assertEqual(blocks1, blocks2)

    # Check that the lists are independent.
    self.assertIsNotNone(blocks1[0])
    blocks1[0] = None
    self.assertIsNotNone(blocks2[0])

    # Check that the elements of the lists are independent.
    self.assertEqual(blocks1[1], blocks2[1])
    blocks1[1].inverse_throughputs[0].inverse_throughput_cycles[:] = [1, 2, 3]
    blocks2[1].inverse_throughputs[0].inverse_throughput_cycles[:] = [4, 5, 6]
    self.assertNotEqual(blocks1[1], blocks2[1])

  def test_get_limited_number_of_blocks(self):
    """Checks that when a limit is given, only that many blocks are returned."""
    num_blocks = 3
    blocks = basic_blocks_with_throughput.get_basic_blocks(num_blocks)

    self.assertLen(blocks, num_blocks)

  def test_cleanup(self):
    num_blocks = 10

    blocks = basic_blocks_with_throughput.get_basic_blocks(
        num_blocks, cleanup_fn=_cleanup)
    self.assertLen(blocks, num_blocks)
    for block in blocks:
      self.assertEmpty(block.inverse_throughputs)

  def test_filter(self):
    num_blocks = 10
    keep_len = 2

    def _block_filter(
        basic_block: throughput_pb2.BasicBlockWithThroughputProto,):
      return len(basic_block.inverse_throughputs) == keep_len

    blocks = basic_blocks_with_throughput.get_basic_blocks(
        num_blocks, keep_fn=_block_filter)
    self.assertLen(blocks, num_blocks)
    for block in blocks:
      self.assertLen(block.inverse_throughputs, keep_len)

  def test_not_enough_blocks(self):
    num_blocks = 1000

    with self.assertRaises(ValueError):
      basic_blocks_with_throughput.get_basic_blocks(num_blocks)


class TestCaseTest(basic_blocks_with_throughput.TestCase, absltest.TestCase):
  num_blocks = 14

  def test_num_loaded_blocks(self):
    self.assertLen(self.blocks_with_throughput, self.num_blocks)
    self.assertLen(self.blocks, self.num_blocks)


if __name__ == '__main__':
  absltest.main()
