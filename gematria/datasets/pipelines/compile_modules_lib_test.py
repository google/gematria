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
import os

from gematria.datasets.pipelines import compile_modules_lib
from gematria.testing.python import ir_utils

from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
import apache_beam as beam


class CompileModulesTests(absltest.TestCase):

  def test_optimize_modules(self):
    ir_string = """
define i32 @a() {
  %a = add i32 3, 5
  ret i32 %a
}
"""
    module_optimizer = compile_modules_lib.OptimizeModules(
        ['default<O0>', 'instcombine']
    )
    optimized_modules = list(
        module_optimizer.process(ir_utils.get_bc_from_ir(ir_string))
    )

    self.assertLen(optimized_modules, 2)

    optimized_ir_string = """
define i32 @a() {
  ret i32 8
}
"""
    optimized_modules_text = [
        ir_utils.get_ir_from_bc(module_bc) for module_bc in optimized_modules
    ]

    self.assertTrue(ir_string in optimized_modules_text[0])
    self.assertTrue(optimized_ir_string in optimized_modules_text[1])

  def test_lowering_get_bbs(self):
    ir_string = """
define i32 @a() {
  ret i32 0
}
"""
    ir_string_bc = ir_utils.get_bc_from_ir(ir_string)

    module_lower_transform = compile_modules_lib.LowerModulesAsm(['-O0', '-O1'])
    lowered_modules = list(module_lower_transform.process(ir_string_bc))

    self.assertLen(lowered_modules, 2)

    get_bbs_transform = compile_modules_lib.GetBBsFromModule()
    bb_hex_values = list(get_bbs_transform.process(lowered_modules[0]))
    self.assertLen(bb_hex_values, 1)
    self.assertEqual(bb_hex_values[0], '31C0C3')

  def test_deduplicate_bbs(self):
    test_bbs = ['aa', 'aa', 'ab', 'ab', 'bc']

    with TestPipeline() as test_pipeline:
      input = test_pipeline | beam.Create(test_bbs)
      output = input | compile_modules_lib.DeduplicateBBs()
      assert_that(output, equal_to(['aa', 'ab', 'bc']))

  def test_get_bbs(self):
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

    test_parquet_file = self.create_tempfile()
    output_file_dir = self.create_tempdir()
    output_file_pattern = os.path.join(output_file_dir, 'bbs')

    ir_utils.create_compile_parquet(
        [ir_string1, ir_string2], test_parquet_file.full_path
    )

    pipeline_constructor = compile_modules_lib.get_bbs(
        test_parquet_file.full_path, output_file_pattern
    )

    with TestPipeline() as test_pipeline:
      pipeline_constructor(test_pipeline)

    with open(output_file_pattern + '-00000-of-00001') as output_file:
      output_file_lines_raw = output_file.readlines()

    output_file_lines = [raw_line.strip() for raw_line in output_file_lines_raw]

    self.assertLen(output_file_lines, 2)
    self.assertContainsSubset(
        ['B801000000C3', 'B802000000C3'], output_file_lines
    )


if __name__ == '__main__':
  absltest.main()