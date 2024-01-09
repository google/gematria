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
        "//gematria/llvm:llvm_to_absl",
        "//gematria/proto:basic_block_cc_proto",
        "//gematria/proto:throughput_cc_proto",
        "//gematria/utils:string",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:MCDisassembler",
        "@llvm-project//llvm:Support",
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

cc_binary(
    name = "find_accessed_addrs_from_bhive",
    srcs = ["find_accessed_addrs_from_bhive.cc"],
    deps = [
        ":bhive_importer",
        ":find_accessed_addrs",
        "//gematria/llvm:canonicalizer",
        "//gematria/llvm:llvm_architecture_support",
        "//gematria/utils:string",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_library(
    name = "find_accessed_addrs",
    srcs = ["find_accessed_addrs.cc"],
    hdrs = ["find_accessed_addrs.h"],
    # This library uses various POSIX APIs. Only tested on Linux, and we'll likely use some
    # Linux-only APIs in future.
    target_compatible_with = [
        "@platforms//os:linux",
    ],
    deps = [
        ":block_wrapper",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "find_accessed_addrs_exegesis",
    srcs = ["find_accessed_addrs_exegesis.cc"],
    hdrs = ["find_accessed_addrs_exegesis.h"],
    deps = [
        ":find_accessed_addrs",
        "//gematria/llvm:disassembler",
        "//gematria/llvm:llvm_architecture_support",
        "@llvm-project//llvm:Exegesis",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "find_accessed_addrs_test",
    srcs = ["find_accessed_addrs_test.cc"],
    # We're testing FindAccessedAddrs, which executes code passed to it. In this test we hardcode
    # some assembled x86 code to execute.
    tags = [
        "not_build:arm",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
    ],
    deps = [
        ":find_accessed_addrs",
        "//gematria/llvm:asm_parser",
        "//gematria/llvm:llvm_architecture_support",
        "//gematria/testing:matchers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/random:seed_sequences",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_test(
    name = "find_accessed_addrs_exegesis_test",
    srcs = ["find_accessed_addrs_exegesis_test.cc"],
    # The llvm-exegesis features used in the exegesis snippet annotator and thus
    # the tests are currently only supported on X86_64.
    tags = [
        "not_build:arm",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
    ],
    deps = [
        ":find_accessed_addrs_exegesis",
        "//gematria/llvm:asm_parser",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "block_wrapper",
    srcs = ["block_wrapper.S"],
    hdrs = ["block_wrapper.h"],
    # This file contains x86_64 assembly that directly invokes Linux syscalls.
    tags = [
        "not_build:arm",
    ],
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    deps = ["@com_google_absl//absl/types:span"],
)
