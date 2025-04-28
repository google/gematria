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
import textwrap

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util as beam_test

from gematria.datasets.pipelines import compile_modules_lib
from gematria.testing.python import ir_utils
from gematria.datasets.python import bhive_to_exegesis
from gematria.proto import execution_annotation_pb2
from gematria.io.python import tfrecord


class CompileModulesTests(parameterized.TestCase):

  def test_optimize_modules(self):
    ir_string = textwrap.dedent("""\
      define i32 @a() {
        %a = add i32 3, 5
        ret i32 %a
      }
    """)

    module_optimizer = compile_modules_lib.OptimizeModules(
        ['default<O0>', 'instcombine']
    )
    module_optimizer.setup()
    optimized_modules = list(
        module_optimizer.process(ir_utils.get_bc_from_ir(ir_string))
    )

    self.assertLen(optimized_modules, 2)

    optimized_ir_string = textwrap.dedent("""\
      define i32 @a() {
        ret i32 8
      }
    """)

    optimized_modules_text = [
        ir_utils.get_ir_from_bc(module_bc) for module_bc in optimized_modules
    ]

    self.assertIn(ir_string, optimized_modules_text[0])
    self.assertIn(optimized_ir_string, optimized_modules_text[1])

  def test_lowering_get_bbs(self):
    ir_string = textwrap.dedent("""\
      define i32 @a() {
        ret i32 0
      }
    """)

    ir_string_bc = ir_utils.get_bc_from_ir(ir_string)

    module_lower_transform = compile_modules_lib.LowerModulesAsm(['-O0', '-O1'])
    module_lower_transform.setup()
    lowered_modules = list(module_lower_transform.process(ir_string_bc))

    self.assertLen(lowered_modules, 2)

    get_bbs_transform = compile_modules_lib.GetBBsFromModule()
    bb_hex_values = list(get_bbs_transform.process(lowered_modules[0]))
    self.assertLen(bb_hex_values, 1)
    self.assertEqual(bb_hex_values[0], '31C0C3')

  def test_deduplicate_values(self):
    test_bbs = ['aa', 'aa', 'ab', 'ab', 'bc']

    with test_pipeline.TestPipeline() as pipeline_under_test:
      input = pipeline_under_test | beam.Create(test_bbs)
      output = input | compile_modules_lib.DeduplicateValues()
      beam_test.assert_that(output, beam_test.equal_to(['aa', 'ab', 'bc']))

  @parameterized.parameters([
      bhive_to_exegesis.AnnotatorType.fast,
      bhive_to_exegesis.AnnotatorType.exegesis,
  ])
  def test_annotate_bbs(self, annotator_type):
    annotator = compile_modules_lib.AnnotateBBs(annotator_type, 50, False)
    annotator.setup()

    annotated_blocks = list(
        annotator.process('4829d38b44246c8b54246848c1fb034829d04839c3')
    )

    self.assertLen(annotated_blocks, 1)

  def test_annotate_bbs_no_loop_register(self):
    annotator = compile_modules_lib.AnnotateBBs(
        bhive_to_exegesis.AnnotatorType.fast, 50, True
    )
    annotator.setup()

    annotated_blocks = list(
        annotator.process('4889C84889DA4889FE4889EC4D89C84D89DA4D89EC4D89FE')
    )

    self.assertLen(annotated_blocks, 0)

  def test_get_vocab(self):
    get_vocab_function = compile_modules_lib.GetVocab()
    get_vocab_function.setup()

    vocab = list(
        get_vocab_function.process('4829d38b44246c8b54246848c1fb034829d04839c3')
    )

    self.assertCountEqual(
        vocab,
        [
            '_MEMORY_',
            'RAX',
            '_IMMEDIATE_',
            'RBX',
            '_D_',
            '_ADDRESS_',
            'EFLAGS',
            'EAX',
            'MOV',
            'EDX',
            'RSP',
            '_NO_REGISTER_',
            'SAR',
            'CMP',
            'RDX',
            'SUB',
            '_DISPLACEMENT_',
        ],
    )

  def test_process_and_filter_bbs(self):
    bb_hex = 'B801000000C3'

    process_and_filter_transform = compile_modules_lib.ProcessAndFilterBBs(
        False
    )
    process_and_filter_transform.setup()

    filtered_bbs = list(process_and_filter_transform.process(bb_hex))

    self.assertLen(filtered_bbs, 1)
    self.assertEqual(filtered_bbs[0], 'B801000000')

    # Check that we return an empty list if we have a BB that contains only
    # instructions that should get filtered.
    bb_hex = 'C3'
    filtered_bbs = list(process_and_filter_transform.process(bb_hex))
    self.assertLen(filtered_bbs, 0)

  @parameterized.parameters([
      bhive_to_exegesis.AnnotatorType.fast,
      bhive_to_exegesis.AnnotatorType.exegesis,
  ])
  def test_get_bbs(self, annotator_type):
    ir_string1 = textwrap.dedent("""\
      define i32 @a() {
        ret i32 1
      }
    """)
    ir_string2 = textwrap.dedent("""\
      define i32 @b() {
        ret i32 2
      }
    """)

    test_parquet_file = self.create_tempfile()
    output_file_dir = self.create_tempdir()
    output_file_pattern = os.path.join(output_file_dir, 'bbs')
    vocab_output_file_pattern = os.path.join(output_file_dir, 'bbvocab')

    ir_utils.create_compile_parquet(
        [ir_string1, ir_string2], test_parquet_file.full_path
    )

    pipeline_constructor = compile_modules_lib.get_bbs(
        test_parquet_file.full_path,
        output_file_pattern,
        False,
        annotator_type,
        50,
        vocab_output_file_pattern,
        False,
    )

    with test_pipeline.TestPipeline() as pipeline_under_test:
      pipeline_constructor(pipeline_under_test)

    block_hex_values = []
    for annotated_block in tfrecord.read_protos(
        [output_file_pattern + '-00000-of-00001'],
        execution_annotation_pb2.BlockWithExecutionAnnotations,
    ):
      block_hex_values.append(annotated_block.block_hex)

    self.assertLen(block_hex_values, 2)
    self.assertContainsSubset(['B801000000', 'B802000000'], block_hex_values)

    with open(
        vocab_output_file_pattern + '-00000-of-00001'
    ) as vocab_file_handle:
      vocab_tokens = [token.strip() for token in vocab_file_handle.readlines()]

    self.assertCountEqual(['_D_', '_IMMEDIATE_', 'MOV', 'EAX'], vocab_tokens)


if __name__ == '__main__':
  absltest.main()
