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
"""A library for recursively copying directories in parallel."""

import functools
import multiprocessing.pool
from os import path

import tensorflow.compat.v1 as tf


class CopyDirError(Exception):
  """An exception raised when copying a directory fails."""


def _reraise(err: Exception):
  raise CopyDirError() from err


def _create_one_dir(dirname: str) -> None:
  tf.io.gfile.makedirs(dirname)


def _copy_one_file(source_and_destination: tuple[str, str],
                   overwrite: bool) -> None:
  from_file, to_file = source_and_destination
  tf.io.gfile.copy(from_file, to_file, overwrite=overwrite)


def copy_dir(from_dir: str,
             to_dir: str,
             overwrite: bool,
             num_workers: int = 10) -> None:
  """Recursively copies files from one directory to another.

  The copying is done in parallel, using tf.io.gfile.copy(). Creates target
  directories as needed; raises an exception when a copy can't be made.

  Args:
    from_dir: The path of the source directory.
    to_dir: The path of the destination directory. Each file and directory
      directly under `from_dir` will appear directly under `to_dir`.
    overwrite: When True, files are overwritten when they exist. When False, it
      is an error when a file already exists in `to_dir`. Note that it is not an
      error when directories in `to_dir` already exist.
    num_workers: The number of threads used for the copying.

  Raises:
    CopyDirError: When copying can't be done.
  """
  # Normalize both directories:
  from_dir = path.abspath(from_dir)
  to_dir = path.abspath(to_dir)

  # Collect the list of input files.
  dirs_to_create = []
  files_to_copy = []

  for dirname, _, filenames in tf.io.gfile.walk(from_dir, onerror=_reraise):
    target_dir = path.join(to_dir, path.relpath(dirname, from_dir))
    dirs_to_create.append(target_dir)
    for filename in filenames:
      from_file = path.join(dirname, filename)
      to_file = path.join(target_dir, filename)
      files_to_copy.append((from_file, to_file))

  pool = multiprocessing.pool.ThreadPool(processes=num_workers)
  pool.map(_create_one_dir, dirs_to_create)
  pool.map(
      functools.partial(_copy_one_file, overwrite=overwrite), files_to_copy)
