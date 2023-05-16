# Copyright 2023 Google Inc.
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

# Make third-party Python libraries imported through Bazel's WORKSPACE system
# visible the same way as if they were imported through PIP or installed as
# a system library, i.e. so that we can write `import sonnet` instead of
# dealing with Bazel's repository paths.
#
# We do this by appending their paths to sys.path; since this init file is in
# the root directory of the project, it will be imported as soon as any
# Gematria module is loaded, and thus safely before importing any third party
# libraries.
#
# The downside of this approach is that we need to maintain the list of third
# party libraries both in `WORKSPACE` and in this file.

import os
import sys


_THIRD_PARTY_REPOS = (
    'graph_nets',
    'pybind11_abseil',
    'sonnet',
)


def _setup_third_party_repos():
  """Add third-party repositories to sys.path.

  The directories are added just before the current directory ("") if it's
  present, or at the end.
  """

  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  third_party_repo_paths = tuple(
      os.path.join(root, repo_name) for repo_name in _THIRD_PARTY_REPOS
  )

  for i, path in enumerate(sys.path):
    if not path:
      sys.path[i:i] = third_party_repo_paths
      break
  else:
    sys.path.extend(third_party_repo_paths)


_setup_third_party_repos()
