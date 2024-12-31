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

from absl.testing import absltest
from apache_beam.testing import test_pipeline

from gematria.datasets.pipelines import benchmark_bbs_lib
from gematria.proto import execution_annotation_pb2
from gematria.io.python import tfrecord
from gematria.datasets.pipelines import benchmark_cpu_scheduler
from gematria.proto import throughput_pb2

BLOCK_FOR_TESTING = execution_annotation_pb2.BlockWithExecutionAnnotations(
    execution_annotations=execution_annotation_pb2.ExecutionAnnotations(
        code_start_address=0,
        block_size=4096,
        block_contents=34359738376,
        accessed_blocks=[86016],
        initial_registers=[
            execution_annotation_pb2.RegisterAndValue(
                register_name='RCX', register_value=86016
            ),
            execution_annotation_pb2.RegisterAndValue(
                register_name='RSI', register_value=86016
            ),
        ],
        loop_register='RAX',
    ),
    block_hex='3b31',
)


class BenchmarkBBsTests(absltest.TestCase):

  def test_benchmark_basic_block(self):
    benchmark_transform = benchmark_bbs_lib.BenchmarkBasicBlock(
        benchmark_cpu_scheduler.BenchmarkSchedulerImplementations.NO_SCHEDULING
    )
    benchmark_transform.setup()

    block_outputs = list(benchmark_transform.process(BLOCK_FOR_TESTING))

    self.assertLen(block_outputs, 1)

    block_hex, block_throughput = block_outputs[0]

    self.assertEqual(block_hex, '3b31')
    self.assertLess(block_throughput, 10)

  def test_serialize_bbs_to_protos(self):
    serialize_transform = benchmark_bbs_lib.SerializeToProto()
    serialize_transform.setup()

    benchmarked_block_data = ('3b31', 5)

    output = list(serialize_transform.process(benchmarked_block_data))
    self.assertLen(output, 1)

  def test_benchmark_bbs(self):
    test_tfrecord = self.create_tempfile()
    tfrecord.write_protos(test_tfrecord.full_path, [BLOCK_FOR_TESTING])

    output_folder = self.create_tempdir()
    output_file_pattern = os.path.join(output_folder, 'bhive-output')

    pipeline_constructor = benchmark_bbs_lib.benchmark_bbs(
        test_tfrecord.full_path,
        output_file_pattern,
        benchmark_cpu_scheduler.BenchmarkSchedulerImplementations.NO_SCHEDULING,
    )

    with test_pipeline.TestPipeline() as pipeline_under_test:
      pipeline_constructor(pipeline_under_test)

    throughputs = []
    for block_with_throughput in tfrecord.read_protos(
        [output_file_pattern + '-00000-of-00001'],
        throughput_pb2.BasicBlockWithThroughputProto,
    ):
      throughputs.append(
          block_with_throughput.inverse_throughputs[
              0
          ].inverse_throughput_cycles[0]
      )

    self.assertLen(throughputs, 1)
    self.assertLess(throughputs[0], 10)


if __name__ == '__main__':
  absltest.main()
