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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/strings",
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
