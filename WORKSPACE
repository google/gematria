workspace(name = "com_google_gematria")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

SKYLIB_VERSION = "1.7.1"

SKYLIB_SHA256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f"

http_archive(
    name = "bazel_skylib",
    sha256 = SKYLIB_SHA256,
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = SKYLIB_VERSION),
        "https://github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = SKYLIB_VERSION),
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

http_archive(
    name = "platforms",
    sha256 = "29742e87275809b5e598dc2f04d86960cc7a55b3067d97221c9abbc9926bff0f",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.11/platforms-0.0.11.tar.gz",
        "https://github.com/bazelbuild/platforms/releases/download/0.0.11/platforms-0.0.11.tar.gz",
    ],
)

http_archive(
    name = "rules_python",
    integrity = "sha256-n587MAqSZOTHeZkxLOZjvl3umlbjYaH2/n7GDhvu+aM=",
    strip_prefix = "rules_python-1.4.1",
    urls = ["https://github.com/bazel-contrib/rules_python/releases/download/1.4.1/rules_python-1.4.1.tar.gz"],
)

http_archive(
    name = "com_google_protobuf",
    integrity = "sha256-a9ncyRsX7yXCat+G23HGfsAkMdyS6Vier4LiKIkjBJY=",
    strip_prefix = "protobuf-29.4",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/refs/tags/v29.4.tar.gz"],
)

git_repository(
    name = "com_google_absl",
    commit = "fb3621f4f897824c0dbe0615fa94543df6192f30",
    remote = "https://github.com/abseil/abseil-cpp.git",
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_toolchains")

py_repositories()

python_register_toolchains(
    name = "python_3_12",
    python_version = "3.12",
)

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pypi",
    python_interpreter_target = "@python_3_12_host//:python",
    requirements_lock = "//:requirements_lock_3_12.txt",
)

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

http_archive(
    name = "pybind11_bazel",
    integrity = "sha256-nfKEMwM2lYyDf7cNw0wKYlTaxSpcmDszc6jCu7eaw14=",
    strip_prefix = "pybind11_bazel-2.13.6",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/v2.13.6.zip"],
)

http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11-BUILD.bazel",
    integrity = "sha256-GtB5JtOHmGyEugbWbsq9VNVZi3kl6rAownjs2Nl8M4U=",
    strip_prefix = "pybind11-2.13.4",
    urls = ["https://github.com/pybind/pybind11/archive/v2.13.4.zip"],
)

http_archive(
    # We can't use the name "pybind11_abseil" for the repository, as this would
    # break the `sys.path` hack that we use for importing third-party Python
    # modules. See the comment under "Python libraries" below for a detailed
    # explanation.
    name = "pybind11_abseil_repo",
    integrity = "sha256-FJaxEuhkFuLc8ohWmj57ZPNTfwsYEyIk9JImbp/3bEQ=",
    strip_prefix = "pybind11_abseil-202402.0",
    urls = ["https://github.com/pybind/pybind11_abseil/releases/download/v202402.0/pybind11_abseil-202402.0.tar.gz"],
)

git_repository(
    name = "com_google_pybind11_protobuf",
    commit = "f02a2b7653bc50eb5119d125842a3870db95d251",
    remote = "https://github.com/pybind/pybind11_protobuf.git",
)

http_archive(
    name = "com_google_googletest",
    integrity = "sha256-eMZ2/GOIFSm/l7+dRZSNkFpmgz+/pTGOos10eMuY85k=",
    strip_prefix = "googletest-1.16.0",
    urls = ["https://github.com/google/googletest/archive/refs/tags/v1.16.0.tar.gz"],
)

# We use a patched version of google/benchmark/BUILD.bazel to keep the name
# used to refer to libpfm4 and its targets consistent with other dependencies.
git_repository(
    name = "com_github_google_benchmark",
    patch_args = ["-p1"],
    patches = ["@//:benchmark_build.patch"],
    remote = "https://github.com/google/benchmark.git",
    tag = "v1.9.1",
)

# rules_foreign_cc is required to build libpfm4 since it is originally
# configured to be built using GNU Make.
git_repository(
    name = "rules_foreign_cc",
    commit = "816905a078773405803e86635def78b61d2f782d",
    remote = "https://github.com/bazelbuild/rules_foreign_cc.git",
)

# Dependencies needed by rules_foreign_cc, in turn needed to build libpfm4.
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

# Used by benchmark to capture metrics using perf counter counters.
git_repository(
    name = "pfm",
    build_file = "@llvm-raw//utils/bazel/third_party_build:pfm.BUILD",
    remote = "https://git.code.sf.net/p/perfmon2/libpfm4",
    tag = "v4.13.0",
)

# We only take the `quipper` sub-package from the `perf_data_converter`
# repository to keep things cleaner.
git_repository(
    name = "com_google_perf_data_converter",
    commit = "442981cd4071fa9b1057b2609406db027e6d6263",
    patch_args = ["-p1"],
    # `strip_prefix` would also work, but this makes generating the patch
    # easier, since `patch_cmds` are applied after the patch itself.
    patch_cmds = [
        "mv ./src/quipper .",
        "rm -rf ./src",
    ],
    patches = ["@//:perf_data_converter.patch"],
    remote = "https://github.com/google/perf_data_converter.git",
)

# `libelf` from `elfutils` and `boringssl` are dependencies for
# `quipper` from `perf_data_converter`.
http_archive(
    name = "elfutils",
    build_file = "//:elfutils.BUILD",
    sha256 = "df76db71366d1d708365fc7a6c60ca48398f14367eb2b8954efc8897147ad871",
    strip_prefix = "elfutils-0.191",
    urls = ["https://sourceware.org/pub/elfutils/0.191/elfutils-0.191.tar.bz2"],
)

http_archive(
    name = "boringssl",
    sha256 = "0a2b7a10fdce3d5ccdc6abf4f5701dca24b97efa75b00d203c50221269605476",
    strip_prefix = "boringssl-ea4425fbb276871cfec5c4e19c12796b3cd1c9ab",
    urls = ["https://github.com/google/boringssl/archive/ea4425fbb276871cfec5c4e19c12796b3cd1c9ab.tar.gz"],
)

# Python libraries

# Bazel copies the contents of these libraries' repositories to the runfiles
# directories of targets that depend on them to make them work as if they were
# imported through PIP or the system package manager. When adding a new Python
# repository here, the name of the repository should be different from the name
# used when importing it in Python, to avoid confusing the module loader; by
# convention, we add `_repo` at the end of the name.

git_repository(
    name = "sonnet_repo",
    commit = "c99b49136210c30fd95bd9c6350fcc3eaf9a72f3",
    patch_args = ["-p1"],
    remote = "https://github.com/deepmind/sonnet.git",
)

new_git_repository(
    name = "graph_nets_repo",
    build_file = "@//:graph_nets.BUILD",
    commit = "adf25162ba21bb0ae176c35483a74fb0c9dff576",
    remote = "https://github.com/deepmind/graph_nets.git",
)

# LLVM and its dependencies

# The pinned version of LLVM, and its SHA256 hash. The `LLVM_COMMIT` variable in
# `.github/workflows/main.yaml` must be updated to match this everytime it is
# changed.
LLVM_COMMIT = "88c4ef2f9fc0cda90c8452bc1c46844aaa722a3e"

LLVM_SHA256 = "163cc60d594aebef8a67088c81d56d19ea90d546de5212b5b1edc1ba4abc662c"

http_archive(
    name = "llvm-raw",
    build_file_content = "# empty",
    sha256 = LLVM_SHA256,
    strip_prefix = "llvm-project-" + LLVM_COMMIT,
    urls = ["https://github.com/llvm/llvm-project/archive/{commit}.zip".format(commit = LLVM_COMMIT)],
)

load(
    "@llvm-raw//utils/bazel:configure.bzl",
    "llvm_configure",
)

llvm_configure(name = "llvm-project")

http_archive(
    name = "llvm_zlib",
    build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
    sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
    strip_prefix = "zlib-ng-2.0.7",
    urls = [
        "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
    ],
)

http_archive(
    name = "llvm_zstd",
    build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
    sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
    strip_prefix = "zstd-1.5.2",
    urls = [
        "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz",
    ],
)

# compile_commands.json extraction for clangd
# https://github.com/hedronvision/bazel-compile-commands-extractor

http_archive(
    name = "hedron_compile_commands",
    sha256 = "3cd0e49f0f4a6d406c1d74b53b7616f5e24f5fd319eafc1bf8eee6e14124d115",
    strip_prefix = "bazel-compile-commands-extractor-3dddf205a1f5cde20faf2444c1757abe0564ff4c",
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/3dddf205a1f5cde20faf2444c1757abe0564ff4c.tar.gz",
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")

hedron_compile_commands_setup()
