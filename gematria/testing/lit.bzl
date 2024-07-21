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

def lit_test(name, data = []):
    lit_file_path = paths.join(native.package_name(), name)

    native.py_test(
        name = name + "_lit_test",
        srcs = ["@llvm-project//llvm:utils/lit/lit.py"],
        args = [lit_file_path, "-vv"],
        main = "@llvm-project//llvm:utils/lit/lit.py",
        data = [":" + name, "lit.cfg.py", "lit.site.cfg.py"] + data,
    )

def glob_lit_tests(name, test_file_exts, data = []):
    test_files = native.glob(
        ["*." + test_file_exts[0]],
    )

    for test_file in test_files:
        lit_test(name = test_file, data = data)

    native.test_suite(
        name = name,
        tests = [t + "_lit_test" for t in test_files],
    )
