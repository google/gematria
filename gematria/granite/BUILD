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

cc_library(
    name = "graph_builder_model_inference",
    srcs = ["graph_builder_model_inference.cc"],
    hdrs = ["graph_builder_model_inference.h"],
    visibility = ["//:external_users"],
    deps = [
        ":graph_builder",
        "//gematria/basic_block",
        "//gematria/model:oov_token_behavior",
        "//gematria/tflite:unsorted_segment_sum_op",
        "//third_party/tensorflow/lite:framework",
        "//third_party/tensorflow/lite/c:c_api_types",
        "//third_party/tensorflow/lite/c:common",
        "//third_party/tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "graph_builder_model_inference_test",
    size = "small",
    srcs = ["graph_builder_model_inference_test.cc"],
    deps = [
        ":graph_builder_model_inference",
        "//file/base:path",
        "//gematria/basic_block",
        "//gematria/basic_block:basic_block_protos",
        "//gematria/testing:parse_proto",
        "//third_party/tensorflow/lite:framework",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:source_location",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "graph_builder_model_inference_main",
    srcs = ["graph_builder_model_inference_main.cc"],
    visibility = ["//:external_users"],
    deps = [
        ":graph_builder_model_inference",
        "//base",
        "//gematria/basic_block",
        "//gematria/basic_block:basic_block_protos",
        "//gematria/datasets:bhive_importer",
        "//gematria/llvm:canonicalizer",
        "//gematria/llvm:llvm_architecture_support",
        "//gematria/proto:basic_block_cc_proto",
        "//third_party/tensorflow/lite:framework_stable",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)
