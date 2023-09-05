load(
    "//:python.bzl",
    "gematria_py_test",
    "gematria_pybind_extension",
)

package(
    default_visibility = ["//visibility:private"],
)

gematria_pybind_extension(
    name = "canonicalizer",
    srcs = ["canonicalizer.cc"],
    pytype_deps = [":llvm_architecture_support"],
    visibility = ["//:internal_users"],
    deps = [
        "//gematria/llvm:canonicalizer",
        "//gematria/llvm:llvm_architecture_support",
    ],
)

gematria_py_test(
    name = "canonicalizer_test",
    size = "small",
    srcs = ["canonicalizer_test.py"],
    deps = [
        ":canonicalizer",
        ":llvm_architecture_support",
    ],
)

gematria_pybind_extension(
    name = "llvm_architecture_support",
    srcs = ["llvm_architecture_support.cc"],
    visibility = ["//:internal_users"],
    deps = [
        "//gematria/llvm:llvm_architecture_support",
        "@pybind11_abseil_repo//pybind11_abseil:status_casters",
    ],
)

gematria_py_test(
    name = "llvm_architecture_support_test",
    size = "small",
    srcs = ["llvm_architecture_support_test.py"],
    deps = [
        ":llvm_architecture_support",
        "//gematria/utils/python:pybind11_abseil_status",
    ],
)
