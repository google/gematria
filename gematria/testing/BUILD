package(
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "parse_proto",
    testonly = True,
    hdrs = ["parse_proto.h"],
    visibility = ["//:internal_users"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "parse_proto_test",
    size = "small",
    srcs = ["parse_proto_test.cc"],
    deps = [
        ":parse_proto",
        "//gematria/proto:canonicalized_instruction_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)
