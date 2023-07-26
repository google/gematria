package(
    default_visibility = ["//visibility:public"],
)

cc_binary(
    name = "llvm-cm",
    srcs = ["llvm-cm/llvm-cm.cpp"],
    deps = [
        "@llvm-project//llvm:AllTargetsAsmParsers",
        "@llvm-project//llvm:AllTargetsCodeGens",
        "@llvm-project//llvm:AllTargetsDisassemblers",
        "@llvm-project//llvm:BinaryFormat",
        "@llvm-project//llvm:DebugInfo",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:MCDisassembler",
        "@llvm-project//llvm:MCParser",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:Support",
    ],
)
