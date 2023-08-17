package(
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "unsorted_segment_sum_op",
    srcs = ["unsorted_segment_sum_op.cc"],
    hdrs = ["unsorted_segment_sum_op.h"],
    visibility = ["//:internal_users"],
    deps = [
        "//third_party/tensorflow/lite/c:c_api_types",
        "//third_party/tensorflow/lite/c:common",
        "//third_party/tensorflow/lite/kernels:kernel_util",
        "//third_party/tensorflow/lite/kernels/internal:tensor",
        "//third_party/tensorflow/lite/kernels/internal:types",
    ],
)

cc_test(
    name = "unsorted_segment_sum_op_test",
    size = "small",
    srcs = ["unsorted_segment_sum_op_test.cc"],
    deps = [
        ":unsorted_segment_sum_op",
        "//third_party/tensorflow/lite:string",
        "//third_party/tensorflow/lite/c:c_api_types",
        "//third_party/tensorflow/lite/kernels:test_util",
        "//third_party/tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)
