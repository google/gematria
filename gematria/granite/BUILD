package(
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "graph_builder",
    srcs = ["graph_builder.cc"],
    hdrs = ["graph_builder.h"],
    visibility = ["//:internal_users"],
    deps = [
        "//gematria/basic_block",
        "//gematria/model:oov_token_behavior",
    ],
)

cc_test(
    name = "graph_builder_test",
    size = "small",
    srcs = ["graph_builder_test.cc"],
    deps = [
        ":graph_builder",
        "//gematria/basic_block",
        "//gematria/basic_block:basic_block_protos",
        "//gematria/model:oov_token_behavior",
        "//gematria/proto:basic_block_cc_proto",
        "//gematria/testing:parse_proto",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

# NOTE(ondrasej): The Granite inference code is built only using CMake due to
# the difficulty of including TFLite as a dependency in a Bazel project.
# TODO(ondrasej): As of 2023-10-09, inference tests are not built or run in the
# open-source code. Add a way to build and run tests in the open-source
# project.
