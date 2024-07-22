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
from rules_python.python.runfiles import runfiles

def _get_llvm_binary_path(tool_name: str) -> str:
  runfiles_dir = os.environ.get('PYTHON_RUNFILES')
  runfiles_env = runfiles.Create({'RUNFILES_DIR': runfiles_dir})
  assert runfiles_env is not None
  return runfiles_env.Rlocation('external/llvm-project/llvm/' + tool_name)

def get_bc_from_ir(ir: str) -> bytes:
  llvm_as_path = _get_llvm_binary_path('llvm-as')
  with subprocess.Popen([llvm_as_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as as_process:
    (output_bc, stderr) = as_process.communicate(ir)
    del stderr
    if as_process.returncode != 0:
      raise ValueError("Expected llvm-as to return 0")
    return output_bc
