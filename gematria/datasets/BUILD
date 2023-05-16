package(
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "bhive_importer",
    srcs = ["bhive_importer.cc"],
    hdrs = ["bhive_importer.h"],
    visibility = ["//:internal_users"],
    deps = [
        "//gematria/basic_block:basic_block_protos",
        "//gematria/llvm:canonicalizer",
        "//gematria/llvm:disassembler",
        "//gematria/proto:basic_block_cc_proto",
        "//gematria/proto:throughput_cc_proto",
        "//gematria/utils:string",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:MCDisassembler",
        "@llvm-project//llvm:Target",
    ],
)

cc_test(
    name = "bhive_importer_test",
    size = "small",
    srcs = ["bhive_importer_test.cc"],
    deps = [
        ":bhive_importer",
        "//gematria/llvm:canonicalizer",
        "//gematria/llvm:llvm_architecture_support",
        "//gematria/testing:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
