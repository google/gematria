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

from absl.testing import absltest
from gematria.testing.python import ir_utils


class IRUtilsTests(absltest.TestCase):

  def test_get_bc_from_ir(self):
    ir_string = """
define i32 @a() {
  ret i32 0
}
"""

    bitcode = ir_utils.get_bc_from_ir(ir_string)
    self.assertTrue(type(bitcode) == bytes)

  def test_get_ir_from_bc(self):
    ir_string = """
define i32 @a() {
  ret i32 0
}
"""
    bitcode = ir_utils.get_bc_from_ir(ir_string)
    ir_roundtrip = ir_utils.get_ir_from_bc(bitcode)
    # Only assert that ir_string is in ir_roundtrip as there will be some
    # additional metadata automatically added to the text.
    self.assertTrue(ir_string in ir_roundtrip)

  def test_create_compile_parquet(self):
    ir_string1 = """
define i32 @a() {
  ret i32 1
}
"""
    ir_string2 = """
define i32 @b() {
  ret i32 2
}
"""
    parquet_file = self.create_tempfile()
    ir_utils.create_compile_parquet(
        [ir_string1, ir_string2], parquet_file.full_path
    )


if __name__ == '__main__':
  absltest.main()
