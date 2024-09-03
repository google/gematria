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

import os

from rules_python.python.runfiles import runfiles


def get_llvm_binary_path(tool_name: str) -> str:
  runfiles_env = runfiles.Create(os.environ)
  assert runfiles_env is not None
  return runfiles_env.Rlocation(os.path.join('llvm-project/llvm', tool_name))
