# Copyright 2022 Google Inc.
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

from os import path

from gematria.io.python import tfrecord
from gematria.proto import canonicalized_instruction_pb2
from gematria.proto import throughput_pb2
from google.protobuf import message
import tensorflow as tf

_CanonicalizedInstructionProto = (
    canonicalized_instruction_pb2.CanonicalizedInstructionProto
)
_CanonicalizedOperandProto = (
    canonicalized_instruction_pb2.CanonicalizedOperandProto
)

_TEST_INSTRUCTIONS = (
    _CanonicalizedInstructionProto(
        mnemonic='MOV',
        input_operands=(
            _CanonicalizedOperandProto(register_name='RAX'),
            _CanonicalizedOperandProto(
                address=_CanonicalizedOperandProto.AddressTuple(
                    base_register='RBX',
                    index_register='RDI',
                    displacement=123,
                    scaling=2,
                    segment='FS',
                )
            ),
        ),
        output_operands=(
            _CanonicalizedOperandProto(
                memory=_CanonicalizedOperandProto.MemoryLocation(
                    alias_group_id=1
                )
            ),
        ),
    ),
    _CanonicalizedInstructionProto(mnemonic='ADD'),
    _CanonicalizedInstructionProto(mnemonic='NOP'),
)


class WriteProtosTest(tf.test.TestCase):

  def _check_write_and_read_again(self, output_filename):
    # Write the test protos to a file.
    tfrecord.write_protos(output_filename, _TEST_INSTRUCTIONS)

    # ...and read them back.
    self.assertTrue(tf.io.gfile.exists(output_filename))
    dataset = tf.data.TFRecordDataset((output_filename,))

    loaded_instructions = []
    for raw_record in dataset:
      loaded_instructions.append(
          _CanonicalizedInstructionProto.FromString(raw_record.numpy())
      )
    self.assertSequenceEqual(loaded_instructions, _TEST_INSTRUCTIONS)

  def test_write_to_a_file(self):
    output_dir = self.create_tempdir()
    output_filename = path.join(output_dir.full_path, 'output.tfrecord')

    self._check_write_and_read_again(output_filename)

  def test_write_to_directories_that_do_not_exist(self):
    output_dir = self.create_tempdir()
    output_filename = path.join(
        output_dir.full_path, 'foo', 'bar', 'output.tfrecord'
    )

    self._check_write_and_read_again(output_filename)

  def test_write_no_protos(self):
    output_dir = self.create_tempdir()
    output_filename = path.join(output_dir.full_path, 'output.tfrecord')

    tfrecord.write_protos(output_filename, ())

    dataset = tf.data.TFRecordDataset((output_filename,))
    for _ in dataset:
      self.fail('There should be no records in the file')


class ReadProtosTest(tf.test.TestCase):

  def test_read_single_file(self):
    input_dir = self.create_tempdir()
    input_filename = path.join(input_dir, 'input.tfrecord')
    tfrecord.write_protos(input_filename, _TEST_INSTRUCTIONS)

    loaded_protos = tuple(
        tfrecord.read_protos(input_filename, _CanonicalizedInstructionProto)
    )
    self.assertSequenceEqual(loaded_protos, _TEST_INSTRUCTIONS)

  def test_read_multiple_files(self):
    # Split the protos into multiple files, one proto per file.
    input_dir = self.create_tempdir()
    input_filenames = []
    for i, proto in enumerate(_TEST_INSTRUCTIONS):
      input_filename = path.join(input_dir.full_path, f'input_{i}.tfrecord')
      input_filenames.append(input_filename)
      tfrecord.write_protos(input_filename, (proto,))

    loaded_protos = tuple(
        tfrecord.read_protos(input_filenames, _CanonicalizedInstructionProto)
    )
    self.assertSequenceEqual(loaded_protos, _TEST_INSTRUCTIONS)

  def test_read_file_that_does_not_exist(self):
    input_dir = self.create_tempdir()
    input_filename = path.join(input_dir.full_path, 'input.tfrecord')

    with self.assertRaises(tf.errors.NotFoundError):
      _ = tuple(
          tfrecord.read_protos(input_filename, _CanonicalizedInstructionProto)
      )

  def test_read_wrong_proto_calss(self):
    input_dir = self.create_tempdir()
    input_filename = path.join(input_dir.full_path, 'input.tfrecord')
    tfrecord.write_protos(input_filename, _TEST_INSTRUCTIONS)

    with self.assertRaises(message.DecodeError):
      _ = tuple(
          tfrecord.read_protos(
              input_filename, throughput_pb2.BasicBlockWithThroughputProto
          )
      )


if __name__ == '__main__':
  tf.test.main()
