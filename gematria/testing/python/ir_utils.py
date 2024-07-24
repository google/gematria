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

import subprocess
import os
import pandas
import pyarrow
from pyarrow import parquet
from rules_python.python.runfiles import runfiles


def _get_llvm_binary_path(tool_name: str) -> str:
  runfiles_env = runfiles.Create(os.environ)
  assert runfiles_env is not None
  return runfiles_env.Rlocation('llvm-project/llvm/' + tool_name)


def get_bc_from_ir(ir: str) -> bytes:
  llvm_as_path = _get_llvm_binary_path('llvm-as')
  with subprocess.Popen(
      [llvm_as_path],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
  ) as as_process:
    (output_bc, stderr) = as_process.communicate(ir.encode('utf-8'))
    del stderr
    if as_process.returncode != 0:
      raise ValueError('Expected llvm-as to return 0')
    return output_bc


def get_ir_from_bc(bc: bytes) -> str:
  llvm_dis_path = _get_llvm_binary_path('llvm-dis')
  with subprocess.Popen(
      [llvm_dis_path],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
  ) as dis_process:
    (output_ir, stderr) = dis_process.communicate(bc)
    del stderr
    if dis_process.returncode != 0:
      raise ValueError('Expected llvm-dis to return 0')
    return output_ir.decode('utf-8')


def create_compile_parquet(ir_examples: str, parquet_path: str) -> None:
  bc_examples = [get_bc_from_ir(ir_example) for ir_example in ir_examples]

  dataframe = pandas.DataFrame.from_dict({'content': bc_examples})

  table = pyarrow.Table.from_pandas(dataframe, preserve_index=False)

  parquet.write_table(table, parquet_path)
