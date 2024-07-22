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

from absl.testing import absltest
from rules_python.python.runfiles import runfiles
import os
from gematria.datasets.python import extract_bbs_from_obj

_ROOT_PATH = 'com_google_gematria'
_OBJECT_FILE_PATH = os.path.join(
    _ROOT_PATH, 'gematria/datasets/python/extract_bbs_from_obj_test_file.o'
)


class ExtractBBsFromObjTests(absltest.TestCase):

  def test_simple_extraction(self):
    runfiles_dir = os.environ.get('PYTHON_RUNFILES')
    runfiles_env = runfiles.Create({'RUNFILES_DIR': runfiles_dir})
    assert runfiles_env is not None
    object_file_path = runfiles_env.Rlocation(_OBJECT_FILE_PATH)
    self.assertTrue(os.path.exists(object_file_path))

    with open(object_file_path, 'rb') as object_file:
      object_file_data = object_file.read()

    basic_blocks = extract_bbs_from_obj.get_basic_block_hex_values(
        object_file_data
    )
    self.assertContainsSubset(['AA', 'BB', 'CC'], basic_blocks)


if __name__ == '__main__':
  absltest.main()
