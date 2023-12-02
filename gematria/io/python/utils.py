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
"""Utilities for reading and writing collections of protos."""

import collections
from collections.abc import Callable, Iterable, Mapping, MutableSequence, Sequence
import re
from typing import Optional, TypeVar

from absl import logging
from gematria.io.python import options
from gematria.proto import throughput_pb2
import numpy as np

T = TypeVar('T')


def _filter_name(filter_fn: Callable[[T], bool]) -> str:
  """Returns a name of a filter function."""
  # Try getting the qualified name of the callable. This works for functions,
  # methods, and lambdas. It does not work for functools.partial and others.
  qual_name = getattr(filter_fn, '__qualname__', None)
  if qual_name:
    return qual_name
  # If there is no qualified name, fall back to repr() that is always available,
  # though not always readable.
  return repr(filter_fn)


def apply_filters(
    items: Iterable[T],
    filters: Sequence[Callable[[T], Optional[T]]],
    max_num_items: int = -1,
) -> Iterable[T]:
  """Applies a sequence of transformers/filters to a stream of items.

  Each transformation/filter is a callback that can return either None to remove
  the item from the stream, a modified item that will replace the input in the
  stream, or the input unchanged.

  The callbacks are applied in the order in which they appear in `filters`; when
  a callback returns None, the transformation sequence is stopped and the
  function moves on to the next item.

  The function stops after successfully yielding `max_num_items` items, or after
  exhausting the input stream.

  Args:
    items: The input stream of items.
    filters: A sequence of transformation/filter functions applied to the items
      from the stream
    max_num_items: The maximal number of items yielded by this function.

  Yields:
    Filtered and transformed items from `items`, in the order in which they
    appeared in the input.
  """
  num_items = 0
  num_visited_items = 0
  num_filtered_protos = collections.defaultdict(int)
  for item in items:
    if max_num_items >= 0 and num_items >= max_num_items:
      return
    if num_visited_items % 10000 == 0:
      logging.info('Processed %d items, kept %d', num_visited_items, num_items)
    num_visited_items += 1
    skip_item = False
    for filter_fn in filters:
      item = filter_fn(item)
      if item is None:
        num_filtered_protos[_filter_name(filter_fn)] += 1
        skip_item = True
        break
    if skip_item:
      continue
    num_items += 1
    yield item
  for filter_name, num_items in num_filtered_protos.items():
    logging.info('Items removed by filter %s: %d', filter_name, num_items)


def _keep_all_throughputs(
    throughputs: MutableSequence[float],
) -> None:
  """Keeps the contents of `throughputs` untouched."""
  del throughputs  # Unused


def _mean_throughput(
    throughputs: MutableSequence[float],
) -> None:
  """Replaces the contents of `throughputs` with its mean value."""
  if not throughputs:
    return
  mean_throughput = np.mean(throughputs)
  throughputs[:] = (mean_throughput,)


def _min_throughput(
    throughputs: MutableSequence[float],
) -> None:
  """Replaces the contents of `throughputs` with its min value."""
  if not throughputs:
    return
  min_throughput = np.min(throughputs)
  throughputs[:] = (min_throughput,)


def _first_throughput(
    throughputs: MutableSequence[float],
) -> None:
  """Replaces the contents of `throughputs` with its first value."""
  if not throughputs:
    return
  del throughputs[1:]


# Mapping from ThroughputSelection values to functions that implement the
# selection.
_THROUGHPUT_SELECTION_FUNCTION: Mapping[
    options.ThroughputSelection,
    Callable[[MutableSequence[float]], None],
] = {
    options.ThroughputSelection.RANDOM: _keep_all_throughputs,
    options.ThroughputSelection.FIRST: _first_throughput,
    options.ThroughputSelection.MEAN: _mean_throughput,
    options.ThroughputSelection.MIN: _min_throughput,
}


def aggregate_throughputs(
    throughput_selection: options.ThroughputSelection,
    block: throughput_pb2.BasicBlockWithThroughputProto,
) -> throughput_pb2.BasicBlockWithThroughputProto:
  """Applies an aggregation function to inverse throughput values in `block`."""
  throughput_filter = _THROUGHPUT_SELECTION_FUNCTION.get(throughput_selection)
  if throughput_filter is None:
    raise ValueError(
        f'Invalid throughput selection strategy: {throughput_selection!r}'
    )
  for throughput in block.inverse_throughputs:
    throughput_filter(throughput.inverse_throughput_cycles)
    for prefix_throughputs in throughput.prefix_inverse_throughputs:
      throughput_filter(prefix_throughputs.inverse_throughput_cycles)
  return block


def _scale_values(
    values: MutableSequence[float], scaling_factor: float
) -> None:
  for i in range(len(values)):
    values[i] *= scaling_factor

def drop_blocks_with_empty_instructions(
    block: throughput_pb2.BasicBlockWithThroughputProto
) -> Optional[throughput_pb2.BasicBlockWithThroughputProto]:
  """Removes basic blocks that do not have any instructions.

  Returns `block` unchanged if it has at least one instruction.

  Args:
    block: The basic block proto to inspect.

  Returns:
    None when `block` has no instructions; otherwise, returns `block`.
  """
  if block.basic_block.canonicalized_instructions:
    return block
  return None

def drop_blocks_with_no_throughputs(
    use_prefixes: bool, block: throughput_pb2.BasicBlockWithThroughputProto
) -> Optional[throughput_pb2.BasicBlockWithThroughputProto]:
  """Removes basic blocks that do not have any inverse throughput values.

  Returns `block` unchanged if it has at least one inverse throughput value for
  at least one task. When `use_prefixes` is False, checks only per-block inverse
  throughputs; when it is True, checks also prefix inverse throughputs.

  Args:
    use_prefixes: Determines whether prefix inverse throughputs are considered.
    block: The basic block proto to inspect.

  Returns:
    None when `block` has no inverse throughputs; otherwise, returns `block`.
  """
  for throughput in block.inverse_throughputs:
    if throughput.inverse_throughput_cycles:
      return block
  if use_prefixes:
    for throughput in block.inverse_throughputs:
      for prefix_throughput in throughput.prefix_inverse_throughputs:
        if prefix_throughput.inverse_throughput_cycles:
          return block
  return None


def scale_throughputs(
    scaling_factor: float, block: throughput_pb2.BasicBlockWithThroughputProto
) -> Optional[throughput_pb2.BasicBlockWithThroughputProto]:
  """Scales the inverse throughputs by scaling_factor in place."""
  for throughput in block.inverse_throughputs:
    _scale_values(throughput.inverse_throughput_cycles, scaling_factor)
    for prefix_throughputs in throughput.prefix_inverse_throughputs:
      _scale_values(
          prefix_throughputs.inverse_throughput_cycles, scaling_factor
      )
  return block


def select_throughputs(
    source_filters: Sequence[re.Pattern[str]],
    block: throughput_pb2.BasicBlockWithThroughputProto,
) -> Optional[throughput_pb2.BasicBlockWithThroughputProto]:
  """Selects inverse throughputs in a block based on provided filters.

  Modifies `block.inverse_throughputs` so that the number of items is equal to
  the number of items in `source_filters`. In the output proto

  Args:
    source_filters: A sequence of compiled regexps to match inverse throughput
      sources that should be kept in the block proto.
    block: The basic block proto to filter; the proto is modified in place and
      returned.

  Returns:
    The block with inverse throughputs corresponding to the filters.
  """
  selected_throughputs = []
  for source_filter in source_filters:
    for throughput in block.inverse_throughputs:
      if source_filter.match(throughput.source):
        selected_throughputs.append(throughput)
        break
    else:
      # If we did not find any throughputs for the filter, we add an empty proto
      # and the value will be masked (or removed) later.
      selected_throughputs.append(throughput_pb2.ThroughputWithSourceProto())
  del block.inverse_throughputs[:]
  block.inverse_throughputs.extend(selected_throughputs)
  return block
