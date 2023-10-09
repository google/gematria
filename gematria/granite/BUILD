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
        "//gematria/utils:string",
        "//third_party/tensorflow/lite:framework",
        "//third_party/tensorflow/lite:type_to_tflitetype",
        "//third_party/tensorflow/lite/c:c_api_types",
        "//third_party/tensorflow/lite/c:common",
        "//third_party/tensorflow/lite/kernels:builtin_ops",
        "@llvm-project//llvm:Support",
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
        "//gematria/llvm:llvm_to_absl",
        "//gematria/testing:parse_proto",
        "//third_party/tensorflow/lite:framework",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:source_location",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "graph_builder_model_inference_main",
    srcs = ["graph_builder_model_inference_main.cc"],
    visibility = ["//:external_users"],
    deps = [
        ":graph_builder_model_inference",
        "//gematria/basic_block",
        "//gematria/llvm:canonicalizer",
        "//gematria/llvm:disassembler",
        "//gematria/llvm:llvm_architecture_support",
        "//gematria/llvm:llvm_to_absl",
        "//gematria/utils:string",
        "//third_party/tensorflow/lite:framework_stable",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Support",
    ],
)
