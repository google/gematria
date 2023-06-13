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

import functools
import re

from absl.testing import absltest
from absl.testing import parameterized
from gematria.io.python import options
from gematria.io.python import utils
from gematria.proto import throughput_pb2


def _filter_multiples(multiplier, value):
  if value % multiplier == 0:
    return value
  return None


def _double(value):
  return 2 * value


class ApplyFiltersTest(absltest.TestCase):

  def test_apply_filters_filter_then_transform(self):
    input_sequence = range(10)
    filters = (functools.partial(_filter_multiples, 2), _double)
    self.assertSequenceEqual(
        tuple(utils.apply_filters(input_sequence, filters)), (0, 4, 8, 12, 16)
    )

  def test_apply_filters_transform_then_filter(self):
    input_sequence = range(10)
    filters = (_double, functools.partial(_filter_multiples, 2))
    self.assertSequenceEqual(
        tuple(utils.apply_filters(input_sequence, filters)),
        (0, 2, 4, 6, 8, 10, 12, 14, 16, 18),
    )

  def test_apply_filters_with_max_num_items(self):
    input_sequence = range(1000)
    filters = (functools.partial(_filter_multiples, 2), _double)
    self.assertSequenceEqual(
        tuple(utils.apply_filters(input_sequence, filters, max_num_items=3)),
        (0, 4, 8),
    )

  def test_apply_multiple_filters(self):
    input_sequence = range(20)
    filters = (
        functools.partial(_filter_multiples, 2),
        functools.partial(_filter_multiples, 3),
    )
    self.assertSequenceEqual(
        tuple(utils.apply_filters(input_sequence, filters)), (0, 6, 12, 18)
    )


# Type alias to make construction of protos in test shorter.
PrefixThroughputProto = (
    throughput_pb2.ThroughputWithSourceProto.PrefixThroughputProto
)


class AggregateThroughputsTest(parameterized.TestCase):
  _ALL_THROUGHPUT_SELECTION_OPTIONS = tuple(
      (selection,) for selection in options.ThroughputSelection
  )

  @parameterized.parameters(*_ALL_THROUGHPUT_SELECTION_OPTIONS)
  def test_with_empty_proto(self, throughput_selection):
    input_proto = throughput_pb2.BasicBlockWithThroughputProto()
    output_proto = utils.aggregate_throughputs(
        throughput_selection, input_proto
    )
    self.assertIs(input_proto, output_proto)
    self.assertEqual(
        output_proto, throughput_pb2.BasicBlockWithThroughputProto()
    )

  def test_keep_random(self):
    # The "randomization" of the throughput value is done at the time of batch
    # scheduling; at the level of loading/filtering, random throughput selection
    # just means that we leave all the values in the proto so that the batch
    # scheduling code can pick one of them.
    input_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake', inverse_throughput_cycles=(1.0, 1.1, 1.2)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge', inverse_throughput_cycles=(3.0, 3.1, 3.2)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell', inverse_throughput_cycles=(2.0, 2.1, 2.2)
            ),
        )
    )
    # utils.aggregate_throughputs() modifies the proto in place. We need to make
    # a copy so that we can compare the output to the original input.
    expected_output_proto = throughput_pb2.BasicBlockWithThroughputProto()
    expected_output_proto.CopyFrom(input_proto)
    output_proto = utils.aggregate_throughputs(
        options.ThroughputSelection.RANDOM, input_proto
    )
    self.assertIs(input_proto, output_proto)
    self.assertEqual(input_proto, expected_output_proto)

  def test_keep_random_with_prefixes(self):
    input_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake',
                inverse_throughput_cycles=(1.0,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(0.3, 0.7)),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge',
                inverse_throughput_cycles=(3.0,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(1.0, 2.0)),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell',
                inverse_throughput_cycles=(2.0,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(0.5, 1.5)),
                ),
            ),
        )
    )
    # utils.aggregate_throughputs() modifies the proto in place. We need to make
    # a copy so that we can compare the output to the original input.
    expected_output_proto = throughput_pb2.BasicBlockWithThroughputProto()
    expected_output_proto.CopyFrom(input_proto)
    output_proto = utils.aggregate_throughputs(
        options.ThroughputSelection.RANDOM, input_proto
    )
    self.assertIs(input_proto, output_proto)
    self.assertEqual(input_proto, expected_output_proto)

  def test_first(self):
    input_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake', inverse_throughput_cycles=(1.0, 1.1, 1.2)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge', inverse_throughput_cycles=(3.0, 3.1, 3.2)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell', inverse_throughput_cycles=(2.0, 2.1, 2.2)
            ),
        )
    )
    expected_output_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake', inverse_throughput_cycles=(1.0,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge', inverse_throughput_cycles=(3.0,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell', inverse_throughput_cycles=(2.0,)
            ),
        )
    )
    output_proto = utils.aggregate_throughputs(
        options.ThroughputSelection.FIRST, input_proto
    )
    self.assertIs(input_proto, output_proto)
    self.assertEqual(input_proto, expected_output_proto)

  def test_first_with_prefixes(self):
    input_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake',
                inverse_throughput_cycles=(1.0, 1.2, 1.3),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(0.3, 0.7)),
                    PrefixThroughputProto(
                        inverse_throughput_cycles=(0.31, 0.71)
                    ),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge',
                inverse_throughput_cycles=(3.0, 3.1, 3.2),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(1.0, 2.0)),
                    PrefixThroughputProto(
                        inverse_throughput_cycles=(1.01, 2.01)
                    ),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell',
                inverse_throughput_cycles=(2.0, 2.1, 2.2),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(0.5, 1.5)),
                    PrefixThroughputProto(
                        inverse_throughput_cycles=(0.51, 1.51)
                    ),
                ),
            ),
        )
    )
    expected_output_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake',
                inverse_throughput_cycles=(1.0,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(0.3,)),
                    PrefixThroughputProto(inverse_throughput_cycles=(0.31,)),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge',
                inverse_throughput_cycles=(3.0,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(1.0,)),
                    PrefixThroughputProto(inverse_throughput_cycles=(1.01,)),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell',
                inverse_throughput_cycles=(2.0,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(0.5,)),
                    PrefixThroughputProto(inverse_throughput_cycles=(0.51,)),
                ),
            ),
        )
    )
    self.assertEqual(
        utils.aggregate_throughputs(
            options.ThroughputSelection.FIRST, input_proto
        ),
        expected_output_proto,
    )

  def test_mean(self):
    input_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake', inverse_throughput_cycles=(10.0, 11.0, 12.0)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge', inverse_throughput_cycles=(30.0, 31.0, 32.0)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell', inverse_throughput_cycles=(20.0, 21.0, 22.0)
            ),
        )
    )
    expected_output_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake', inverse_throughput_cycles=(11.0,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge', inverse_throughput_cycles=(31.0,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell', inverse_throughput_cycles=(21.0,)
            ),
        )
    )
    output_proto = utils.aggregate_throughputs(
        options.ThroughputSelection.MEAN, input_proto
    )
    self.assertIs(input_proto, output_proto)
    self.assertEqual(
        input_proto,
        expected_output_proto,
        msg=f'Expected:\n{expected_output_proto}\nActual:\n{output_proto}',
    )

  def test_mean_with_prefixes(self):
    input_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake',
                inverse_throughput_cycles=(1.0, 2.0, 3.0),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(1.0, 2.0)),
                    PrefixThroughputProto(
                        inverse_throughput_cycles=(2.0, 3.0, 1.0)
                    ),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge',
                inverse_throughput_cycles=(30.0, 40.0, 50.0),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(
                        inverse_throughput_cycles=(10.0, 20.0)
                    ),
                    PrefixThroughputProto(
                        inverse_throughput_cycles=(15.0, 16.0, 17.0)
                    ),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell',
                inverse_throughput_cycles=(20.0, 21.0, 22.0),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(
                        inverse_throughput_cycles=(5.0, 15.0)
                    ),
                    PrefixThroughputProto(
                        inverse_throughput_cycles=(5.0, 6.0, 7.0)
                    ),
                ),
            ),
        )
    )
    expected_output_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake',
                inverse_throughput_cycles=(2.0,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(1.5,)),
                    PrefixThroughputProto(inverse_throughput_cycles=(2.0,)),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge',
                inverse_throughput_cycles=(40.0,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(15.0,)),
                    PrefixThroughputProto(inverse_throughput_cycles=(16.0,)),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell',
                inverse_throughput_cycles=(21.0,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(10.0,)),
                    PrefixThroughputProto(inverse_throughput_cycles=(6.0,)),
                ),
            ),
        )
    )
    output_proto = utils.aggregate_throughputs(
        options.ThroughputSelection.MEAN, input_proto
    )
    self.assertEqual(
        output_proto,
        expected_output_proto,
        msg=f'Expected:\n{expected_output_proto}\nActual:\n{output_proto}',
    )

  def test_min(self):
    input_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake', inverse_throughput_cycles=(1.5, 1.1, 1.2)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge', inverse_throughput_cycles=(3.0, 3.1, 3.2)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell', inverse_throughput_cycles=(2.7, 2.1, 2.2)
            ),
        )
    )
    expected_output_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake', inverse_throughput_cycles=(1.1,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge', inverse_throughput_cycles=(3.0,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell', inverse_throughput_cycles=(2.1,)
            ),
        )
    )
    output_proto = utils.aggregate_throughputs(
        options.ThroughputSelection.MIN, input_proto
    )
    self.assertIs(input_proto, output_proto)
    self.assertEqual(
        input_proto,
        expected_output_proto,
        msg=f'Expected:\n{expected_output_proto}\nActual:\n{output_proto}',
    )

  def test_min_with_prefixes(self):
    input_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake',
                inverse_throughput_cycles=(1.5, 1.2, 1.3),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(0.3, 0.7)),
                    PrefixThroughputProto(
                        inverse_throughput_cycles=(0.731, 0.71)
                    ),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge',
                inverse_throughput_cycles=(3.5, 3.4, 3.2),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(1.0, 2.0)),
                    PrefixThroughputProto(
                        inverse_throughput_cycles=(3.01, 2.01)
                    ),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell',
                inverse_throughput_cycles=(2.0, 2.1, 2.2),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(0.5, 1.5)),
                    PrefixThroughputProto(
                        inverse_throughput_cycles=(0.51, 1.51)
                    ),
                ),
            ),
        )
    )
    expected_output_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake',
                inverse_throughput_cycles=(1.2,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(0.3,)),
                    PrefixThroughputProto(inverse_throughput_cycles=(0.71,)),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge',
                inverse_throughput_cycles=(3.2,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(1.0,)),
                    PrefixThroughputProto(inverse_throughput_cycles=(2.01,)),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell',
                inverse_throughput_cycles=(2.0,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(0.5,)),
                    PrefixThroughputProto(inverse_throughput_cycles=(0.51,)),
                ),
            ),
        )
    )
    output_proto = utils.aggregate_throughputs(
        options.ThroughputSelection.MIN, input_proto
    )
    self.assertIs(input_proto, output_proto)
    self.assertEqual(
        output_proto,
        expected_output_proto,
        msg=f'Expected:\n{expected_output_proto}\nActual:\n{output_proto}',
    )


class DropBlocksWithNoThroughputsTest(absltest.TestCase):
  pass


class ScaleThroughputs(absltest.TestCase):

  def test_with_empty_proto(self):
    input_proto = throughput_pb2.BasicBlockWithThroughputProto()
    output_proto = utils.scale_throughputs(1.1, input_proto)
    self.assertIs(input_proto, output_proto)
    self.assertEqual(
        output_proto, throughput_pb2.BasicBlockWithThroughputProto()
    )

  def test_with_throughputs(self):
    input_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake', inverse_throughput_cycles=(1.0,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge', inverse_throughput_cycles=(3.0,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell', inverse_throughput_cycles=(2.0,)
            ),
        )
    )
    expected_output_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake', inverse_throughput_cycles=(1.5,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge', inverse_throughput_cycles=(4.5,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell', inverse_throughput_cycles=(3.0,)
            ),
        )
    )
    output_proto = utils.scale_throughputs(1.5, input_proto)
    self.assertEqual(
        output_proto,
        expected_output_proto,
        f'Expected:\n{expected_output_proto}\n\nActual:\n{output_proto}',
    )

  def test_with_prefix_throughputs(self):
    input_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake',
                inverse_throughput_cycles=(1.0,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(0.3, 0.7)),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge',
                inverse_throughput_cycles=(3.0,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(1.0, 2.0)),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell',
                inverse_throughput_cycles=(2.0,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(0.5, 1.5)),
                ),
            ),
        )
    )
    expected_output_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake',
                inverse_throughput_cycles=(2.0,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(0.6, 1.4)),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge',
                inverse_throughput_cycles=(6.0,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(2.0, 4.0)),
                ),
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell',
                inverse_throughput_cycles=(4.0,),
                prefix_inverse_throughputs=(
                    PrefixThroughputProto(inverse_throughput_cycles=(1.0, 3.0)),
                ),
            ),
        )
    )
    output_proto = utils.scale_throughputs(2.0, input_proto)
    self.assertEqual(
        output_proto,
        expected_output_proto,
        f'Expected:\n{expected_output_proto}\n\nActual:\n{output_proto}',
    )


class SelectThroughputsTest(absltest.TestCase):
  _SOURCE_FILTERS = (
      re.compile('ivybridge'),
      re.compile('haswell'),
      re.compile('skylake'),
  )

  def test_with_empty_proto(self):
    input_proto = throughput_pb2.BasicBlockWithThroughputProto()
    expected_output_proto = utils.select_throughputs(
        self._SOURCE_FILTERS, input_proto
    )
    self.assertIs(input_proto, expected_output_proto)
    self.assertEqual(
        expected_output_proto,
        throughput_pb2.BasicBlockWithThroughputProto(
            inverse_throughputs=(
                throughput_pb2.ThroughputWithSourceProto(),
                throughput_pb2.ThroughputWithSourceProto(),
                throughput_pb2.ThroughputWithSourceProto(),
            )
        ),
    )

  def test_reorder_throughputs(self):
    input_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake', inverse_throughput_cycles=(1.0,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge', inverse_throughput_cycles=(3.0,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell', inverse_throughput_cycles=(2.0,)
            ),
        )
    )
    expected_output_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='ivybridge', inverse_throughput_cycles=(3.0,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell', inverse_throughput_cycles=(2.0,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake', inverse_throughput_cycles=(1.0,)
            ),
        )
    )
    self.assertEqual(
        expected_output_proto,
        utils.select_throughputs(self._SOURCE_FILTERS, input_proto),
    )

  def test_pick_from_multiple_variants(self):
    input_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake', inverse_throughput_cycles=(1.0,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake', inverse_throughput_cycles=(3.0,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake', inverse_throughput_cycles=(2.0,)
            ),
        )
    )
    expected_output_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(),
            throughput_pb2.ThroughputWithSourceProto(),
            throughput_pb2.ThroughputWithSourceProto(
                source='skylake', inverse_throughput_cycles=(1.0,)
            ),
        )
    )
    self.assertEqual(
        expected_output_proto,
        utils.select_throughputs(self._SOURCE_FILTERS, input_proto),
    )

  def test_drop_unused_sources(self):
    input_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell', inverse_throughput_cycles=(1.0,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='foobar', inverse_throughput_cycles=(3.0,)
            ),
            throughput_pb2.ThroughputWithSourceProto(
                source='foobar2', inverse_throughput_cycles=(2.0,)
            ),
        )
    )
    expected_output_proto = throughput_pb2.BasicBlockWithThroughputProto(
        inverse_throughputs=(
            throughput_pb2.ThroughputWithSourceProto(),
            throughput_pb2.ThroughputWithSourceProto(
                source='haswell', inverse_throughput_cycles=(1.0,)
            ),
            throughput_pb2.ThroughputWithSourceProto(),
        )
    )
    self.assertEqual(
        expected_output_proto,
        utils.select_throughputs(self._SOURCE_FILTERS, input_proto),
    )


if __name__ == '__main__':
  absltest.main()
