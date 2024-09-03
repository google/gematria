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
"""Utility functions for writing tests that work with IR.

This library contains several functions that perform functions such as converting
between LLVM Bitcode and textural IR to enable more easily writing tests that
work with these formats.
"""

import subprocess
import os
from collections.abc import Sequence


import pandas
import pyarrow
from pyarrow import parquet

import gematria.utils.python.runfiles


def get_bc_from_ir(ir: str) -> bytes:
  """Generates bitcode for an IR string.

  Takes an IR string as input and outputs the bitcode representation of the
  input textual IR as bytes.

  Args:
    ir: A string containing the textual IR to process.

  Returns:
    Bytes containing the bitcode representation of the input IR.

  Raises:
    CalledProcessError: If running llvm-as fails.
  """
  llvm_as_path = gematria.utils.python.runfiles.get_llvm_binary_path('llvm-as')
  return subprocess.run(
      [llvm_as_path], input=ir.encode('utf-8'), capture_output=True, check=True
  ).stdout


def get_ir_from_bc(bc: bytes) -> str:
  """Converts bitcode to textual IR.

  Takes LLVM bitcode in the form of bytes and converts it into textual IR.

  Args:
    bc: The bitcode as bytes.

  Returns:
    A string containing the textual IR from the bitcode.

  Raises:
    CalledProcessError: If running llvm-dis fails.
  """
  llvm_dis_path = gematria.utils.python.runfiles.get_llvm_binary_path(
      'llvm-dis'
  )
  return subprocess.run(
      [llvm_dis_path], input=bc, capture_output=True, check=True
  ).stdout.decode('utf-8')


def create_compile_parquet(
    ir_examples: Sequence[str], parquet_path: str
) -> None:
  """Creates a test parquet file matching the format of the ComPile dataset.

  Creates a parquet file containing LLVM modules in the form of bitcode from
  the ir_examples passed in. The parquet file is in the same format as the
  ComPile dataset, in particular storing the bitcode in the content column.

  Args:
    ir_examples: A list of IR strings that should be included in the parquet
      file in the form of bitcode.
    parquet_path: The path to place the output parquet file at.
  """
  bc_examples = [get_bc_from_ir(ir_example) for ir_example in ir_examples]

  dataframe = pandas.DataFrame.from_dict({'content': bc_examples})

  table = pyarrow.Table.from_pandas(dataframe, preserve_index=False)

  parquet.write_table(table, parquet_path)
