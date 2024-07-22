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
"""Contains rules for running lit tests."""

load("@bazel_skylib//lib:paths.bzl", "paths")

def _lit_test(test_file_name, data = []):
    """Runs an individual lit test.

    Runs an individual lit test using the llvm-lit runner
    and automatically passes along the lit configuration files.

    Args:
      test_file_name: The name of the test file to generate the test target
        for.
      data: An array of additional data dependencies needed by the test.
    """
    lit_file_path = paths.join(native.package_name(), test_file_name)

    native.py_test(
        name = test_file_name + "_lit_test",
        srcs = ["@llvm-project//llvm:utils/lit/lit.py"],
        args = [lit_file_path, "-vv"],
        main = "@llvm-project//llvm:utils/lit/lit.py",
        data = [":" + test_file_name, "lit.cfg.py", "lit.site.cfg.py"] + data,
    )

def glob_lit_tests(name, test_file_exts, data = []):
    """Creates test targets for all lit tests in a directory.

    Globs a directory for files starting with the extensions listed in
    test_file_exts, generates test targets for all individual files, and
    creates a test suite containing all the files.

    Args:
      name: The name of the test suite target.
      test_file_exts: The file extensions of the test files to search for.
      data: An array of additional data dependencies needed by the test.
    """
    test_files = native.glob(
        ["*." + test_file_ext for test_file_ext in test_file_exts],
    )

    for test_file in test_files:
        _lit_test(test_file_name = test_file, data = data)

    native.test_suite(
        name = name,
        tests = [t + "_lit_test" for t in test_files],
    )
