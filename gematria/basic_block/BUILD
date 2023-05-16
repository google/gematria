package(
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "basic_block",
    srcs = ["basic_block.cc"],
    hdrs = ["basic_block.h"],
    visibility = ["//:internal_users"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "basic_block_test",
    size = "small",
    srcs = ["basic_block_test.cc"],
    deps = [
        ":basic_block",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "basic_block_protos",
    srcs = ["basic_block_protos.cc"],
    hdrs = ["basic_block_protos.h"],
    visibility = ["//:internal_users"],
    deps = [
        ":basic_block",
        "//gematria/proto:basic_block_cc_proto",
        "//gematria/proto:canonicalized_instruction_cc_proto",
        "@com_google_protobuf//:protobuf_lite",
    ],
)

cc_test(
    name = "basic_block_protos_test",
    size = "small",
    srcs = ["basic_block_protos_test.cc"],
    deps = [
        ":basic_block",
        ":basic_block_protos",
        "//gematria/proto:basic_block_cc_proto",
        "//gematria/proto:canonicalized_instruction_cc_proto",
        "//gematria/testing:matchers",
        "//gematria/testing:parse_proto",
        "@com_google_googletest//:gtest_main",
    ],
)
