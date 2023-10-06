package(
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "asm_parser",
    srcs = ["asm_parser.cc"],
    hdrs = ["asm_parser.h"],
    visibility = ["//:internal_users"],
    deps = [
        ":diagnostics",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:BinaryFormat",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:MCParser",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_test(
    name = "asm_parser_test",
    size = "small",
    srcs = ["asm_parser_test.cc"],
    deps = [
        ":asm_parser",
        ":llvm_architecture_support",
        "//gematria/testing:llvm",
        "//gematria/testing:matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:X86UtilsAndDesc",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "canonicalizer",
    srcs = ["canonicalizer.cc"],
    hdrs = ["canonicalizer.h"],
    visibility = ["//:external_users"],
    deps = [
        "//gematria/basic_block",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86UtilsAndDesc",
    ],
)

cc_test(
    name = "canonicalizer_test",
    size = "small",
    srcs = ["canonicalizer_test.cc"],
    deps = [
        ":asm_parser",
        ":canonicalizer",
        ":llvm_architecture_support",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "diagnostics",
    hdrs = ["diagnostics.h"],
    visibility = ["//:internal_users"],
    deps = [
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "diagnostics_test",
    size = "small",
    srcs = ["diagnostics_test.cc"],
    deps = [
        ":diagnostics",
        ":llvm_architecture_support",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "disassembler",
    srcs = ["disassembler.cc"],
    hdrs = ["disassembler.h"],
    visibility = ["//:internal_users"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:MCDisassembler",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "disassembler_test",
    size = "small",
    srcs = ["disassembler_test.cc"],
    deps = [
        ":disassembler",
        ":llvm_architecture_support",
        "//gematria/testing:llvm",
        "//gematria/testing:matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:X86UtilsAndDesc",
    ],
)

cc_library(
    name = "llvm_architecture_support",
    srcs = ["llvm_architecture_support.cc"],
    hdrs = ["llvm_architecture_support.h"],
    visibility = ["//:internal_users"],
    deps = [
        ":llvm_target_x86",
        "@com_google_absl//absl/log:check",
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
    name = "llvm_architecture_support_test",
    size = "small",
    srcs = ["llvm_architecture_support_test.cc"],
    deps = [
        ":llvm_architecture_support",
        "//gematria/testing:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "llvm_target_x86",
    deps = [
        "@llvm-project//llvm:X86AsmParser",
        "@llvm-project//llvm:X86CodeGen",
        "@llvm-project//llvm:X86Disassembler",
        "@llvm-project//llvm:X86Info",
    ],
    alwayslink = 1,
)
