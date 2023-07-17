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
"""Contains portable rules for generating protocol buffer libraries."""

load("@rules_cc//cc:defs.bzl", _cc_proto_library = "cc_proto_library")
load("@com_google_protobuf//:protobuf.bzl", _py_proto_library = "py_proto_library")

load("//third_party/bazel_rules/rules_python/python:proto.bzl", "py_proto_library")

def gematria_proto_library(name = None, srcs = None, deps = (), **kwargs):
    """Creates proto library target and language bindings.

    Assumes that `name` has the format `{base_name}_proto`, and creates the following targets:
      - proto_library(name = name),
      - cc_proto_library(name = f"{base_name}_cc_proto"),
      - py_proto_library(name = f"{base_name}_py_pb2").
    Smooths over the inconsistencies of the open-source py_proto_library() rules.

    Args:
        name: The name of the proto library; used for the underlying proto_library() target.
        srcs: The source .proto file(s).
        deps: The list of proto libraries that the proto depends on. The targets in the list should
            be either names passed to other gematria_proto_library() invocations, or names of
            proto_library() targets that have C++ and Python bindings following the naming shceme
            used by this function.
        **kwargs: All other arguments are passed to all targets created by this function.
    """

    if not name.endswith("_proto"):
        fail("Proto library name must end with _proto")
    name_base = name[:-6]
    cc_name = name_base + "_cc_proto"
    py_name = name_base + "_py_pb2"
    binding_deps = [":" + name]
    native.proto_library(
        name = name,
        srcs = srcs,
        deps = deps,
        **kwargs
    )
    _cc_proto_library(
        name = cc_name,
        deps = binding_deps,
        **kwargs
    )
    py_deps = [target[:-6] + "_py_pb2" for target in deps]
    _py_proto_library(
        name = py_name,
        srcs = srcs,
        deps = py_deps,
        **kwargs
    )
