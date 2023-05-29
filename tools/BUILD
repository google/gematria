load(
    "//:python.bzl",
    "gematria_py_binary",
)

package(
    default_visibility = ["//visibility:private"],
)

PACKAGE_SHARED_OBJECT_DEPS = [
    "//gematria/llvm/python:canonicalizer.so",
    "//gematria/llvm/python:llvm_architecture_support.so",
    "//gematria/basic_block/python:basic_block_protos.so",
    "//gematria/basic_block/python:basic_block.so",
    "//gematria/datasets/python:bhive_importer.so",
    "//gematria/granite/python:graph_builder.so",
    "//gematria/model/python:oov_token_behavior.so",
    # Use a custom target rather than the default external one for the pybind11 status
    # shared object as the default target puts it in a place that setuptools doesn't
    # understand.
    "//:package_pybind11_abseil_status",
]

PACKAGE_PYTHON_DEPS = [
    "//gematria/proto:basic_block_py_pb2",
    "//gematria/proto:canonicalized_instruction_py_pb2",
    "//gematria/proto:throughput_py_pb2",
]

genrule(
    name = "shared_object_list",
    srcs = PACKAGE_SHARED_OBJECT_DEPS,
    outs = ["shared_object_list.txt"],
    cmd = "echo {} > $@".format(" ".join(["$(rootpath {})".format(x) for x in PACKAGE_SHARED_OBJECT_DEPS])),
)

gematria_py_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.py"],
    data = PACKAGE_SHARED_OBJECT_DEPS + PACKAGE_PYTHON_DEPS + [
        "//:requirements.in",
        "//tools:shared_object_list",
    ],
    visibility = ["//visibility:public"],
)
