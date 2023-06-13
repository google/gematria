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
"""Helper functions for working with protos stored in .tfrecord files."""

from collections.abc import Iterable, Sequence
from os import path
from typing import Type, TypeVar

import tensorflow.compat.v1 as tf

from google.protobuf import message

Proto = TypeVar('Proto', bound=message.Message)


def read_protos(filenames: Sequence[str],
                proto_class: Type[Proto]) -> Iterable[Proto]:
  """Reads protos of type `Proto` from `filenames`.

  Assumes that `filenames` is a list of files that each contains serialized
  protos of type `Proto`.

  Args:
    filenames: A single file name or a list of file names to read.
    proto_class: The class of the protos to parse from the files.

  Yields:
    The parsed protos.

  Raises:
    tf.errors.OpError: On input/output errors.
    DecodeError: When a record in the input files can't be parsed as `Proto`.
  """
  if isinstance(filenames, str):
    # NOTE(ondrasej): In Python, `str` is also an `Iterable[str]` (it iterates
    # over all characters of the string). Since the type checker would not stop
    # us when passing a single file name instead of a collection, we just fix it
    # and do what the user expects.
    filenames = (filenames,)
  for filename in filenames:
    for raw_record in tf.io.tf_record_iterator(filename):
      yield proto_class.FromString(raw_record)


def write_protos(filename: str, protos: Iterable[Proto]) -> None:
  """Writes all protos from `protos` to a single .tfrecord file.

  Uses the binary serialization (via proto.SerializeToString()) when writing the
  data.

  Args:
    filename: The name of the file to write to.
    protos: A collection of protos to write to. This collection is iterated over
      only once.

  Raises:
    tf.errors.OpError: On input/output errors.
  """
  output_dir = path.dirname(filename)
  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)
  with tf.io.TFRecordWriter(filename) as writer:
    for proto in protos:
      writer.write(proto.SerializeToString())
