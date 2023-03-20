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

from absl import logging
import tensorflow.compat.v1 as tf

from gematria.io.python import gfile_copy


def _create_a_file_with_dirs(filename):
  logging.info('Creating directory %s', path.dirname(filename))
  tf.io.gfile.makedirs(path.dirname(path.abspath(filename)))
  with tf.io.gfile.GFile(filename, 'w') as f:
    logging.info('Writing to file %s', filename)
    f.write(filename)


def _enumerate_files_in_dir(top):
  files = set()
  for dirname, _, filenames in tf.io.gfile.walk(top):
    for filename in filenames:
      files.add(path.relpath(path.join(dirname, filename), top))
  return sorted(files)


class CopyDirTest(tf.test.TestCase):

  def test_copy_some_files(self):
    filenames = (
        'foo/bar/baz.txt',
        'foo/bar.txt',
        'foo/foo/foo/bar.txt',
    )
    # Create files from `filenames` under `from_dir`.
    from_dir = self.create_tempdir('from')
    for filename in filenames:
      _create_a_file_with_dirs(path.join(from_dir.full_path, filename))

    # Create an empty `to_dir`.
    to_dir = self.create_tempdir('to')

    # Run the code under test.
    gfile_copy.copy_dir(from_dir.full_path, to_dir.full_path, overwrite=False)

    # Check that each file was correctly copied with the contents from
    # `from_dir`.
    for filename in filenames:
      dst_file = path.join(to_dir.full_path, filename)
      self.assertTrue(
          tf.io.gfile.exists(dst_file), f'File does not exist: {dst_file}'
      )
      with tf.io.gfile.GFile(dst_file, 'r') as f:
        contents = f.read()
      self.assertEqual(contents, path.join(from_dir.full_path, filename))

    # Check that only the files from `filenames` were copied.
    self.assertCountEqual(_enumerate_files_in_dir(to_dir.full_path), filenames)

  def test_overwrite(self):
    # Create `from_dir` and `to_dir` both containing a file `foo.txt`.
    from_dir = self.create_tempdir('from')
    from_dir_file = path.join(from_dir.full_path, 'foo.txt')
    _create_a_file_with_dirs(from_dir_file)

    to_dir = self.create_tempdir('to')
    to_dir_file = path.join(to_dir.full_path, 'foo.txt')
    _create_a_file_with_dirs(to_dir_file)

    # Run the code under test.
    gfile_copy.copy_dir(from_dir.full_path, to_dir.full_path, overwrite=True)

    # Check that the file in `to_dir` was overwritten with the contents of the
    # `from_dir` file.
    with tf.io.gfile.GFile(to_dir_file, 'r') as f:
      contents = f.read()
    self.assertEqual(contents, from_dir_file)

  def test_do_not_overwrite(self):
    # Create `from_dir` and `to_dir` both containing `foo.txt`.
    from_dir = self.create_tempdir('from')
    _create_a_file_with_dirs(path.join(from_dir.full_path, 'foo.txt'))

    to_dir = self.create_tempdir('to')
    _create_a_file_with_dirs(path.join(to_dir.full_path, 'foo.txt'))

    # Check that copying fails because of overwriting.
    with self.assertRaises(tf.errors.OpError):
      gfile_copy.copy_dir(from_dir.full_path, to_dir.full_path, overwrite=False)


if __name__ == '__main__':
  tf.test.main()
