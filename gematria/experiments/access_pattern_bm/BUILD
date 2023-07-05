package(
    default_visibility = ["//visibility:private"],
)

config_setting(
    name = "balance_flushing_time",
    define_values = {
        "balance_flushing_time": "true",
    },
)

FEATURE_OPTS = ["-mclflushopt"]

cc_library(
    name = "linked_list",
    srcs = ["linked_list.cc"],
    hdrs = ["linked_list.h"],
    copts = FEATURE_OPTS,
)

cc_test(
    name = "linked_list_bm",
    size = "small",
    srcs = ["linked_list_test.cc"],
    copts = select({
        ":balance_flushing_time": ["-DBALANCE_FLUSHING_TIME"],
        "//conditions:default": [],
    }),
    deps = [
        ":linked_list",
        "@com_github_google_benchmark//:benchmark_main",
    ],
)

cc_library(
    name = "contiguous_matrix",
    srcs = ["contiguous_matrix.cc"],
    hdrs = ["contiguous_matrix.h"],
    copts = FEATURE_OPTS,
    deps = ["@com_google_absl//absl/base:core_headers"],
)

cc_test(
    name = "contiguous_matrix_bm",
    size = "small",
    srcs = ["contiguous_matrix_test.cc"],
    copts = select({
        ":balance_flushing_time": ["-DBALANCE_FLUSHING_TIME"],
        "//conditions:default": [],
    }),
    deps = [
        ":contiguous_matrix",
        "@com_github_google_benchmark//:benchmark_main",
    ],
)

cc_library(
    name = "vec_of_vec_matrix",
    srcs = ["vec_of_vec_matrix.cc"],
    hdrs = ["vec_of_vec_matrix.h"],
    copts = FEATURE_OPTS,
)

cc_test(
    name = "vec_of_vec_matrix_bm",
    size = "small",
    srcs = ["vec_of_vec_matrix_test.cc"],
    copts = select({
        ":balance_flushing_time": ["-DBALANCE_FLUSHING_TIME"],
        "//conditions:default": [],
    }),
    deps = [
        ":vec_of_vec_matrix",
        "@com_github_google_benchmark//:benchmark_main",
    ],
)

cc_library(
    name = "stl_container",
    srcs = ["stl_container.cc"],
    hdrs = ["stl_container.h"],
    copts = FEATURE_OPTS,
)

cc_test(
    name = "stl_container_bm",
    size = "small",
    srcs = ["stl_container_test.cc"],
    copts = select({
        ":balance_flushing_time": ["-DBALANCE_FLUSHING_TIME"],
        "//conditions:default": [],
    }) + FEATURE_OPTS,  # Since templated functions using _mm_clflushopt are defined in the header
    deps = [
        ":stl_container",
        "@com_github_google_benchmark//:benchmark_main",
    ],
)

cc_library(
    name = "stl_assoc_container",
    srcs = ["stl_assoc_container.cc"],
    hdrs = ["stl_assoc_container.h"],
    copts = FEATURE_OPTS,
)

cc_test(
    name = "stl_assoc_container_bm",
    size = "small",
    srcs = ["stl_assoc_container_test.cc"],
    copts = select({
        ":balance_flushing_time": ["-DBALANCE_FLUSHING_TIME"],
        "//conditions:default": [],
    }) + FEATURE_OPTS,  # Since templated functions using _mm_clflushopt are defined in the header
    deps = [
        ":stl_assoc_container",
        "@com_github_google_benchmark//:benchmark_main",
    ],
)
