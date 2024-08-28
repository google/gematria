load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make")

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
)

configure_make(
    name = "libelf",
    configure_options = [
        "--disable-debuginfod",
        "--disable-libdebuginfod",
    ],
    copts = ["-Wno-error"],
    lib_source = ":all_files",
    out_lib_dir = "",
    out_shared_libs = ["libelf.so"],
    targets = [
        "PREFIX=$$INSTALLDIR$$ " +
        "install",
    ],
    visibility = ["//visibility:public"],
)
