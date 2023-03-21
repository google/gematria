workspace(name = "com_google_gematria")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

git_repository(
    name = "com_google_protobuf",
    remote = "https://github.com/protocolbuffers/protobuf.git",
    # TODO(ondrasej): This is the maximal version compatible with
    # TensorFlow 2.11.0 (last version on PyPI as of 2023-03-03).
    tag = "v19.5",
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

# Required for protobuf to build.
bind(
    name = "python_headers",
    actual = "@com_google_protobuf//util/python:python_headers",
)

git_repository(
    name = "com_google_absl",
    remote = "https://github.com/abseil/abseil-cpp.git",
    tag = "20230125.0",
)

git_repository(
    name = "org_mizux_bazelpybind11",
    commit = "cebceda1061d0a1eb5692a5cae615012f4bba228",
    remote = "https://github.com/Mizux/bazel-pybind11.git",
)

git_repository(
    name = "com_google_googletest",
    remote = "https://github.com/google/googletest.git",
    tag = "release-1.12.1",
)

git_repository(
    name = "rules_proto",
    remote = "https://github.com/bazelbuild/rules_proto.git",
    tag = "5.3.0-21.7",
)

# Python
git_repository(
    name = "rules_python",
    remote = "https://github.com/bazelbuild/rules_python.git",
    tag = "0.19.0",
)

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pypi",
    requirements_lock = "//:requirements.txt",
)

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

git_repository(
    name = "pybind11_bazel",
    commit = "faf56fb3df11287f26dbc66fdedf60a2fc2c6631",
    patch_args = ["-p1"],
    patches = ["@org_mizux_bazelpybind11//patches:pybind11_bazel.patch"],
    remote = "https://github.com/pybind/pybind11_bazel.git",
)

new_git_repository(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    remote = "https://github.com/pybind/pybind11.git",
    tag = "v2.10.3",
)

git_repository(
    name = "com_google_pybind11_protobuf",
    commit = "6128db54998e52f4003c0e04f472aba7a3dbe835",
    remote = "https://github.com/pybind/pybind11_protobuf.git",
)

load("@pybind11_bazel//:python_configure.bzl", _pybind11_python_configure = "python_configure")

_pybind11_python_configure(
    name = "local_config_python",
    python_version = "3",
)

# Python libraries

# We need to manipulate sys.path to make these libraries work as if they were imported
# through PIP or the system package manager. When adding a new Python repository here,
# 1. the name of the repository should be different from the name used when importing it
#    in Python, to avoid confusing the module loader; by convention, we add `_repo` at
#    the end of the name.
# 2. the names of the repositories must be added to the list of third-party repositories
#    in `gematria/__init__.py` to make sure that they are added to sys.path.

git_repository(
    name = "sonnet_repo",
    commit = "cd5b5fa48e15e4d020f744968f5209949ebe750f",
    patch_args = ["-p1"],
    patches = ["//:sonnet.patch"],
    remote = "https://github.com/deepmind/sonnet.git",
)

new_git_repository(
    name = "graph_nets_repo",
    build_file = "@//:graph_nets.BUILD",
    commit = "adf25162ba21bb0ae176c35483a74fb0c9dff576",
    remote = "https://github.com/deepmind/graph_nets.git",
)
