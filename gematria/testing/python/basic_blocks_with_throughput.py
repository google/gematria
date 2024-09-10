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
"""A small set of basic blocks with throughputs for testing the ML models."""

from collections.abc import Callable, Iterable, Sequence
import copy
import itertools
import os
import unittest

from gematria.basic_block.python import basic_block
from gematria.basic_block.python import throughput
from gematria.basic_block.python import throughput_protos
from gematria.basic_block.python import tokens
from gematria.proto import throughput_pb2
from google.protobuf import text_format
from rules_python.python.runfiles import runfiles

# The path to the basic blocks in text format in the resources of the test.
_ROOT_PATH = 'com_google_gematria'
_BASIC_BLOCK_RESOURCE_PATH = os.path.join(
    _ROOT_PATH, 'gematria/testing/testdata/basic_blocks_with_throughput.pbtxt'
)
_ANNOTATED_BASIC_BLOCK_RESOURCE_PATH = os.path.join(
    _ROOT_PATH,
    'gematria/testing/testdata/annotated_basic_blocks_with_throughput.pbtxt',
)
# Parsed basic block. An exception is thrown if the basic blocks do not parse.
_BASIC_BLOCKS: throughput_pb2.BasicBlockWithThroughputListProto | None = None

CleanupFn = Callable[
    [throughput_pb2.BasicBlockWithThroughputProto],
    throughput_pb2.BasicBlockWithThroughputProto,
]
KeepFn = Callable[[throughput_pb2.BasicBlockWithThroughputProto], bool]


def _get_basic_block_list_proto(get_annotated_blocks=False):
  """Loads basic blocks from test data."""
  global _BASIC_BLOCKS
  if _BASIC_BLOCKS is None:
    runfiles_dir = os.environ.get('PYTHON_RUNFILES')
    runfiles_env = runfiles.Create({'RUNFILES_DIR': runfiles_dir})
    assert runfiles_env is not None
    with open(
        runfiles_env.Rlocation(
            _ANNOTATED_BASIC_BLOCK_RESOURCE_PATH
            if get_annotated_blocks
            else _BASIC_BLOCK_RESOURCE_PATH
        ),
        'rt',
    ) as f:
      _BASIC_BLOCKS = text_format.Parse(
          f.read(), throughput_pb2.BasicBlockWithThroughputListProto()
      )
  return _BASIC_BLOCKS


def get_basic_blocks(
    num_blocks: int = 0,
    get_annotated_blocks: bool = False,
    cleanup_fn: CleanupFn | None = None,
    keep_fn: KeepFn | None = None,
) -> list[throughput_pb2.BasicBlockWithThroughputProto]:
  """Returns a small collection of basic blocks with throughput and used tokens.

  Each call to the function returns a new deep copy of the data. Any
  modifications made to the list or the protos in it do not propagate to other
  tests.

  Args:
    num_blocks: The number of blocks to return. When zero, all blocks are
      returned.
    get_annotated_blocks: Whether the returned blocks should contain instruction
      annotations or not. Defaults to False.
    cleanup_fn: An optional function that cleans up the proto before it is
      filtered via keep_fn.
    keep_fn: A function that returns True for basic blocks that should be
      included in the results, and False for protos that should be skipped. When
      no function is provided, all protos are included.

  Returns:
    A list of blocks and a list of tokens for canonicalized representations.
  """
  source_blocks = _get_basic_block_list_proto(get_annotated_blocks).basic_blocks
  num_blocks = num_blocks or len(source_blocks)
  # The following makes a deep copy of all the basic block protos, and ensures
  # that the container is a list rather than a repeated proto field container.
  all_blocks = (copy.deepcopy(block) for block in source_blocks)
  if cleanup_fn:
    all_blocks = map(cleanup_fn, all_blocks)
  if keep_fn:
    all_blocks = filter(keep_fn, all_blocks)

  blocks = list(itertools.islice(all_blocks, num_blocks))

  if len(blocks) < num_blocks:
    raise ValueError(
        f'Not enough blocks in test data. Found: {len(blocks)}, '
        f'expected: {num_blocks}.'
    )

  return blocks


def _get_block_tokens(
    blocks: Iterable[basic_block.BasicBlock],
) -> Sequence[str]:
  """Returns a sorted list of tokens for `blocks`.

  The returned list contains all tokens that appear in `blocks`, and all
  structural tokens from `tokens`. The returned list is sorted, and it contains
  each token only once.

  Args:
    blocks: A collection of basic blocks from which the tokens are extracted.
  """
  unique_tokens = set(tokens.STRUCTURAL_TOKENS)
  for block in blocks:
    for instruction in block.instructions:
      unique_tokens.update(instruction.as_token_list())
  return sorted(unique_tokens)


# NOTE(ondrasej): The inheritance is not necessary, we add it mainly to make
# type checkers happy.
class TestCase(unittest.TestCase):
  """Provides example basic blocks for use inside unit tests.

  This class can be used as a mixin with one of the classes that derive from
  unittest.TestCase, e.g. absltest.TestCase or tf.test.TestCase.

  Child classes can control the number of basic blocks loaded from test data by
  overriding the value of num_blocks on the child class.

  Attributes:
    blocks: Only the basic block part of blocks_with_throughput.
    annotated_blocks: Variant of `blocks` with instruction annotations.
    block_protos: Basic block with throughput protos loaded from test data.
    annotated_block_protos: Variant of `block_protos` with instruction
      annotations.
    blocks_with_throughput: The basic blocks with throughput loaded from the
      test data. Initializes by the setUp() overload of this class.
    annotated_blocks_with_throughput: Variant of `blocks_with_throughput` with
      instruction annotations.
    num_blocks: The number of blocks retrieved from the test data. By default,
      we use just a single basic block as it is enough to test the whole
      learning process, and all the algorithms should be able to overfit on it
      easily. Child classes can override this by setting `self.num_blocks`
      before calling `super().setUp()` in their setUp() method.
    keep_function: A function that specifies what blocks to keep when collecting
      test data. This is intended for filtering for specific blocks that
      exercise specific functionality for regression tests.
    tokens: The list of all tokens appearing in self.blocks. The tokens are
      sorted, and each token appears in the list only once.
    annotation_names: The list of annotation names to be used.
  """

  num_blocks: int = 1
  keep_function: KeepFn = None

  blocks: list[basic_block.BasicBlock]
  annotated_blocks: list[basic_block.BasicBlock]
  block_protos: list[throughput_pb2.BasicBlockWithThroughputProto]
  annotated_block_protos: list[throughput_pb2.BasicBlockWithThroughputProto]
  blocks_with_throughput: list[throughput.BasicBlockWithThroughput]
  annotated_blocks_with_throughput: list[throughput.BasicBlockWithThroughput]

  tokens: Sequence[str]
  annotation_names: Sequence[str]

  def setUp(self):
    super().setUp()

    self.block_protos = get_basic_blocks(
        self.num_blocks, keep_fn=self.keep_function
    )
    self.annotated_block_protos = get_basic_blocks(
        self.num_blocks, get_annotated_blocks=True, keep_fn=self.keep_function
    )
    self.assertLen(self.block_protos, self.num_blocks)
    self.assertLen(self.annotated_block_protos, self.num_blocks)

    self.blocks_with_throughput = [
        throughput_protos.block_with_throughput_from_proto(proto)
        for proto in self.block_protos
    ]
    self.annotated_blocks_with_throughput = [
        throughput_protos.block_with_throughput_from_proto(proto)
        for proto in self.annotated_block_protos
    ]
    self.blocks = [
        block_with_throughput.block
        for block_with_throughput in self.blocks_with_throughput
    ]
    self.annotated_blocks = [
        block_with_throughput.block
        for block_with_throughput in self.annotated_blocks_with_throughput
    ]
    self.tokens = _get_block_tokens(self.blocks)
    self.annotation_names = [
        'made_up_cache_miss_freq',
    ]
