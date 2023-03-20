load("//:python.bzl", "gematria_py_library", "gematria_py_test")
load(
    "@pypi//:requirements.bzl",
    "requirement",
)

package(
    default_visibility = ["//visibility:private"],
)

gematria_py_library(
    name = "flag_utils",
    srcs = ["flag_utils.py"],
    visibility = ["//:internal_users"],
)

gematria_py_test(
    name = "flag_utils_test",
    size = "small",
    srcs = ["flag_utils_test.py"],
    deps = [
        ":flag_utils",
    ],
)

gematria_py_library(
    name = "timer",
    srcs = ["timer.py"],
    visibility = ["//:internal_users"],
    deps = [requirement("absl-py")],
)

gematria_py_test(
    name = "timer_test",
    size = "small",
    srcs = ["timer_test.py"],
    deps = [
        ":timer",
    ],
)
