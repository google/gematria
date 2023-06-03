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

r"""Creates a Gematria data set from a BHive data set.

Reads basic blocks and throughput data from a BHive CSV file, and writes them in
the proto format to a Gematria .tfrecord file.

Usage:
  import_from_bhive \
      --gematria_input_csv=/tmp/bhive/skl.csv \
      --gematria_output_tfrecord=/tmp/bhive/skl.tfrecord \
      --gematria_throughput_source_name="bhive: skl"
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
from gematria.datasets.python import bhive_importer
from gematria.llvm.python import canonicalizer
from gematria.llvm.python import llvm_architecture_support
from pybind11_abseil import status
import tensorflow as tf


_INPUT_CSV_FILE = flags.DEFINE_string(
    'gematria_input_csv',
    None,
    'The name of the BHive CSV file to import',
    required=True,
)
_OUTPUT_TFRECORD_FILE = flags.DEFINE_string(
    'gematria_output_tfrecord',
    None,
    'The name of the TFRecord file to write the data to.',
    required=True,
)
_SOURCE_NAME = flags.DEFINE_string(
    'gematria_throughput_source_name',
    None,
    'The name of the throughput source used for the throughput data from the'
    ' CSV file.',
    required=True,
)
_THROUGHPUT_SCALING = flags.DEFINE_float(
    'gematria_throughput_scaling',
    1.0,
    'The scaling coefficient applied to the throughput values from the CSV'
    ' file.',
)
_LLVM_TRIPLE = flags.DEFINE_string(
    'gematria_llvm_triple',
    'x86_64',
    'The LLVM triple used for disassembling the instructions in the data set.',
)
_MACHINE_CODE_HEX_COLUMN_INDEX = flags.DEFINE_integer(
    'machine_code_hex_column_index',
    '0',
    'The index of the the machine code hex column in the input CSV file.',
)
_THROUGHPUT_COLUMN_INDEX = flags.DEFINE_integer(
    'throughput_column_index',
    '1',
    'The index of the throughput value column in the input CSV file.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  try:
    llvm = llvm_architecture_support.LlvmArchitectureSupport.from_triple(
        _LLVM_TRIPLE.value
    )
  except status.StatusNotOk:
    logging.exception(
        'LLVM triple "%s" is not known or supported.', _LLVM_TRIPLE.value
    )
    return

  # TODO(ondrasej): Update this so that the canonicalizer is created using the
  # LLVM triple. As of 2023-05, this is OK, because we support only x86-64
  # anyway.
  canonicalizer_obj = canonicalizer.Canonicalizer.x86_64(llvm)
  importer = bhive_importer.BHiveImporter(canonicalizer_obj)

  with (
      tf.io.gfile.GFile(_INPUT_CSV_FILE.value, 'r') as bhive_csv_file,
      tf.io.TFRecordWriter(_OUTPUT_TFRECORD_FILE.value) as writer,
  ):
    num_input_blocks = 0
    num_skipped_blocks = 0
    for line in bhive_csv_file:
      if num_input_blocks % 1000 == 0:
        logging.info(
            'Processed %d blocks, skipped %d.',
            num_input_blocks,
            num_skipped_blocks,
        )
      num_input_blocks += 1
      try:
        block_proto = importer.basic_block_with_throughput_proto_from_csv_line(
            source_name=_SOURCE_NAME.value,
            line=line,
            machine_code_hex_column_index=_MACHINE_CODE_HEX_COLUMN_INDEX.value,
            throughput_column_index=_THROUGHPUT_COLUMN_INDEX,
            throughput_scaling=_THROUGHPUT_SCALING.value,
        )
      except status.StatusNotOk:
        logging.exception('Could not process line "%s"', line)
        num_skipped_blocks += 1
        continue

      writer.write(block_proto.SerializeToString())


if __name__ == '__main__':
  app.run(main)
