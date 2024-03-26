load("//:python.bzl", "gematria_py_library", "gematria_py_test")

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
    deps = [
    ],
)

gematria_py_test(
    name = "timer_test",
    size = "small",
    srcs = ["timer_test.py"],
    deps = [
        ":timer",
    ],
)

# The following is a dummy helper rule that allows us to use the pybind11-generated
# Python library as a regualar Python dependency throughout the code base.
gematria_py_library(
    name = "pybind11_abseil_status",
    data = [
        "@pybind11_abseil_repo//pybind11_abseil:status.so",
    ],
    visibility = ["//:internal_users"],
)
