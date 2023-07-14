package(
    default_visibility = ["//visibility:private"],
)

# The macro BALANCE_FLUSHING_TIME is defined when this matches. This enables the
# lines of code that create and flush mock data structures in *_NoFlush
# benchmarks to cancel out the time spent doing those operations as compared to
# the corresponding *_Flush benchmarks.
config_setting(
    name = "balance_flushing_time",
    define_values = {
        "balance_flushing_time": "true",
    },
)

COMMON_TEST_HDRS = ["configuration.h"]

FEATURE_OPTS = ["-mclflushopt"]

BALANCE_FLUSHING_TIME_OPTS = select({
    ":balance_flushing_time": ["-DGEMATRIA_EXPERIMENTS_ACCESS_PATTERN_BM_BALANCE_FLUSHING_TIME"],
    "//conditions:default": [],
})

cc_library(
    name = "linked_list",
    srcs = ["linked_list.cc"],
    hdrs = ["linked_list.h"],
    copts = FEATURE_OPTS,
)

cc_test(
    name = "linked_list_test",
    size = "small",
    srcs = COMMON_TEST_HDRS + [
        "linked_list.h",
        "linked_list_test.cc",
    ],
    copts = BALANCE_FLUSHING_TIME_OPTS,
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
    name = "contiguous_matrix_test",
    size = "small",
    srcs = COMMON_TEST_HDRS + [
        "contiguous_matrix.h",
        "contiguous_matrix_test.cc",
    ],
    copts = BALANCE_FLUSHING_TIME_OPTS,
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
    deps = ["@com_google_absl//absl/base:core_headers"],
)

cc_test(
    name = "vec_of_vec_matrix_test",
    size = "small",
    srcs = COMMON_TEST_HDRS + [
        "vec_of_vec_matrix.h",
        "vec_of_vec_matrix_test.cc",
    ],
    copts = BALANCE_FLUSHING_TIME_OPTS,
    deps = [
        ":vec_of_vec_matrix",
        "@com_github_google_benchmark//:benchmark_main",
    ],
)

cc_test(
    name = "stl_container_test",
    size = "small",
    srcs = COMMON_TEST_HDRS + [
        "stl_container.h",
        "stl_container_test.cc",
    ],
    copts = BALANCE_FLUSHING_TIME_OPTS + FEATURE_OPTS,  # Since templated functions using _mm_clflushopt are defined in the header
    deps = [
        "@com_github_google_benchmark//:benchmark_main",
    ],
)

cc_test(
    name = "stl_assoc_container_test",
    size = "small",
    srcs = COMMON_TEST_HDRS + [
        "stl_assoc_container.h",
        "stl_assoc_container_test.cc",
    ],
    copts = BALANCE_FLUSHING_TIME_OPTS + FEATURE_OPTS,  # Since templated functions using _mm_clflushopt are defined in the header
    deps = [
        "@com_github_google_benchmark//:benchmark_main",
    ],
)
