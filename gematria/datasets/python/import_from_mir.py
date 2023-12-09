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

r"""Creates a Gematria data set from a MIR data set.

Reads basic blocks and throughput data from a directory containing MIR files and perf measurement, and writes them in
the proto format to a Gematria .tfrecord file.

Usage:
  import_from_mir \
      --gematria_input_dir=/tmp/skl/mirs \
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


_INPUT_DIR = flags.DEFINE_string(
    'gematria_input_dir',
    None,
    'The name of directory containing all raw MIR files with performance throughput',
    required=True,
)

_INPUT_DIR2 = flags.DEFINE_string(
    'gematria_input_dir2',
    None,
    'The name of directory containing all raw MIR files with performance throughput',
)

_MODEL_TYPE = flags.DEFINE_string(
    'gematria_model_format',
    None,
    'The format of dataset to be imported. [NO_LIVE_INFO, PER_BB_LIVE_INFO, PER_FUNC_LIVE_INFO]',
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
    100.0,
    'The scaling coefficient applied to the throughput values from the CSV'
    ' file.',
)
_LLVM_TRIPLE = flags.DEFINE_string(
    'gematria_llvm_triple',
    'x86_64',
    'The LLVM triple used for disassembling the instructions in the data set.',
)
_MACHINE_BASIC_BLOCK_NAME_COLUMN_INDEX = flags.DEFINE_integer(
    'machine_basic_block_name_column_index',
    '0',
    'The index of the the machine code hex column in the input CSV file.',
)
_MACHINE_HEX_COLUMN_INDEX = flags.DEFINE_integer(
    'machine_hex_column_index',
    '1',
    'The index of the the machine code hex column in the input CSV file.',
)
_THROUGHPUT_COLUMN_INDEX = flags.DEFINE_integer(
    'throughput_column_index',
    '2',
    'The index of the throughput value column in the input CSV file.',
)


@flags.multi_flags_validator(
    [_MACHINE_BASIC_BLOCK_NAME_COLUMN_INDEX.name, _THROUGHPUT_COLUMN_INDEX.name],
    message=(
        'Expected machine code column and throughput column indices to be'
        ' different'
    ),
)
def _validate_input_columns(flags_dict):
  return (
      flags_dict[_MACHINE_BASIC_BLOCK_NAME_COLUMN_INDEX.name]
      != flags_dict[_THROUGHPUT_COLUMN_INDEX.name]
  )


import os
from gematria.datasets.python import bhive_importer
from gematria.llvm.python import canonicalizer
from gematria.llvm.python import llvm_architecture_support
from pybind11_abseil import status
import tensorflow as tf

def is_mode_interference_graph(model_type):
    return model_type == "PER_BB_LIVE_INFO" or model_type == "PER_FUNC_LIVE_INFO"


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
  if is_mode_interference_graph(_MODEL_TYPE.value):
    logging.info('Creating BHiveImporter with interference graph %s', _MODEL_TYPE.value)
    importer = bhive_importer.BHiveImporter(canonicalizer_obj, _MODEL_TYPE.value)
  else:
    importer = bhive_importer.BHiveImporter(canonicalizer_obj)
  
  with (
     tf.io.TFRecordWriter(_OUTPUT_TFRECORD_FILE.value) as writer,
  ):
    num_input_blocks = 0
    num_input_files = 0
    num_skipped_blocks = 0
    num_skipped_files = 0
    input_dirs = [_INPUT_DIR.value]
    if _INPUT_DIR2.value:
        input_dirs.append(_INPUT_DIR2.value)
    for input_dir in input_dirs:
        for filename in os.listdir(input_dir):
            if filename.endswith(".mir"):
                if num_input_files % 1000 == 0:
                    logging.info(
                        'Processed %d files, skipped %d.',
                        num_input_files,
                        num_skipped_files,
                    )
                mir_file = os.path.join(input_dir, filename)
                print("mir file is " + mir_file)
                perf_file = os.path.join(input_dir, filename.replace(".mir", ".perf"))
                liveinfo_file = os.path.join(input_dir, filename + ".liveinfo")
                try:
                    # load the MIR file
                    logging.info('Procssing %s file', mir_file)
                    importer.LoadMIRModule(mir_file)
                    logging.info('Loading live info %s file', liveinfo_file)
                    # if is interference graph, then we need to load the liveinfo file
                    importer.parse_interference_graph(liveinfo_file)
                    num_input_files += 1
                    # iterate over each line in the corresponding .perf file
                    with tf.io.gfile.GFile(perf_file, 'r') as bhive_csv_file:
                        for line in bhive_csv_file:
                            if num_input_blocks % 1000 == 0:
                                logging.info(
                                    'Processed %d blocks, skipped %d.',
                                    num_input_blocks,
                                    num_skipped_blocks,
                                )
                            num_input_blocks += 1
                            try:
                                hex = line.split(",")[_MACHINE_HEX_COLUMN_INDEX.value]
                                BB_name = line.split(",")[_MACHINE_BASIC_BLOCK_NAME_COLUMN_INDEX.value]
                                through_put = line.split(",")[_THROUGHPUT_COLUMN_INDEX.value]
                                # skip blocks with throughput -1
                                if float(through_put) == -1 or float(through_put) < 0.1 or float(through_put) > 10:
                                    num_skipped_blocks += 1
                                    continue
                                block_proto = importer.ParseMIRCsvLine(
                                    source_name=_SOURCE_NAME.value,
                                    line=line.strip(),
                                    BB_name_index = _MACHINE_BASIC_BLOCK_NAME_COLUMN_INDEX.value,
                                    throughput_column_index = _THROUGHPUT_COLUMN_INDEX.value,
                                    throughput_scaling=_THROUGHPUT_SCALING.value,
                                )
                                writer.write(block_proto.SerializeToString())
                            except:
                                num_skipped_blocks += 1
                except:
                    logging.exception('Could not load file "%s"', mir_file)
                    num_skipped_files += 1
    logging.info(
        'Processed %d files, skipped %d.',
        num_input_files,
        num_skipped_files,
    )
    logging.info(
        'Processed %d blocks, skipped %d.',
        num_input_blocks,
        num_skipped_blocks,
    )


if __name__ == '__main__':
  app.run(main)
