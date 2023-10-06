package(
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "string",
    srcs = ["string.cc"],
    hdrs = ["string.h"],
    visibility = ["//:internal_users"],
    deps = [
    ],
)

cc_test(
    name = "string_test",
    size = "small",
    srcs = ["string_test.cc"],
    deps = [
        ":string",
        "@com_google_googletest//:gtest_main",
    ],
)
