load(
    "//:python.bzl",
    "gematria_py_binary",
    "gematria_py_test",
    "gematria_pybind_extension",
)

package(
    default_visibility = ["//visibility:private"],
)

gematria_pybind_extension(
    name = "bhive_importer",
    srcs = ["bhive_importer.cc"],
    py_deps = [
        "//gematria/llvm/python:canonicalizer",
        "//gematria/proto:basic_block_py_pb2",
        "//gematria/proto:canonicalized_instruction_py_pb2",
        "//gematria/proto:throughput_py_pb2",
    ],
    visibility = ["//:internal_users"],
    deps = [
        "//gematria/basic_block:basic_block_protos",
        "//gematria/datasets:bhive_importer",
        "//gematria/llvm:canonicalizer",
        "@com_google_pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        "@llvm-project//llvm:Support",
        "@pybind11_abseil_repo//pybind11_abseil:status_casters",
    ],
)

gematria_py_test(
    name = "bhive_importer_test",
    size = "small",
    srcs = ["bhive_importer_test.py"],
    deps = [
        ":bhive_importer",
        "//gematria/llvm/python:canonicalizer",
        "//gematria/llvm/python:llvm_architecture_support",
        "//gematria/proto:basic_block_py_pb2",
        "//gematria/proto:canonicalized_instruction_py_pb2",
        "//gematria/proto:throughput_py_pb2",
    ],
)

gematria_py_binary(
    name = "import_from_bhive",
    srcs = ["import_from_bhive.py"],
    deps = [
        ":bhive_importer",
        "//gematria/llvm/python:canonicalizer",
        "//gematria/llvm/python:llvm_architecture_support",
        "//gematria/utils/python:pybind11_abseil_status",
    ],
)

gematria_py_binary(
    name = "import_from_mir",
    srcs = ["import_from_mir.py"],
    deps = [
        ":bhive_importer",
        "//gematria/llvm/python:canonicalizer",
        "//gematria/llvm/python:llvm_architecture_support",
        "//gematria/utils/python:pybind11_abseil_status",
    ],
)
