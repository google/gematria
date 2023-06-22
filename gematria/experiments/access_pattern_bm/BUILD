package(
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "linked_list",
    srcs = ["linked_list.cc"],
    hdrs = ["linked_list.h"],
    copts = ["-mclflushopt"],
    visibility = ["//:internal_users"],
)

cc_test(
    name = "linked_list_bm",
    size = "small",
    srcs = ["linked_list_bm.cc"],
    deps = [
        ":linked_list",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_library(
    name = "contiguous_matrix",
    srcs = ["contiguous_matrix.cc"],
    hdrs = ["contiguous_matrix.h"],
    copts = ["-mclflushopt"],
    visibility = ["//:internal_users"],
)

cc_test(
    name = "contiguous_matrix_bm",
    size = "small",
    srcs = ["contiguous_matrix_bm.cc"],
    deps = [
        ":contiguous_matrix",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_library(
    name = "vec_of_vec_matrix",
    srcs = ["vec_of_vec_matrix.cc"],
    hdrs = ["vec_of_vec_matrix.h"],
    copts = ["-mclflushopt"],
    visibility = ["//:internal_users"],
)

cc_test(
    name = "vec_of_vec_matrix_bm",
    size = "small",
    srcs = ["vec_of_vec_matrix_bm.cc"],
    deps = [
        ":vec_of_vec_matrix",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_library(
    name = "stl_container",
    srcs = ["stl_container.cc"],
    hdrs = ["stl_container.h"],
    copts = ["-mclflushopt"],
    visibility = ["//:internal_users"],
)

cc_test(
    name = "stl_container_bm",
    size = "small",
    srcs = ["stl_container_bm.cc"],
    copts = ["-mclflushopt"],  # Since templated functions using _mm_clflushopt are defined in the header
    deps = [
        ":stl_container",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_library(
    name = "stl_assoc_container",
    srcs = ["stl_assoc_container.cc"],
    hdrs = ["stl_assoc_container.h"],
    copts = ["-mclflushopt"],
    visibility = ["//:internal_users"],
)

cc_test(
    name = "stl_assoc_container_bm",
    size = "small",
    srcs = ["stl_assoc_container_bm.cc"],
    copts = ["-mclflushopt"],  # Since templated functions using _mm_clflushopt are defined in the header
    deps = [
        ":stl_assoc_container",
        "@com_google_benchmark//:benchmark",
    ],
)
