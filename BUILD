load("@rules_python//python:pip.bzl", "compile_pip_requirements")

package(
    default_visibility = ["//visibility:private"],
)

exports_files(["LICENSE"])

package_group(
    name = "external_users",
    includes = [":internal_users"],
    packages = [],
)

# A package group used for internal visibility within the EXEgesis code.
package_group(
    name = "internal_users",
    packages = [
        # Gematria code base.
        "//gematria/...",
    ],
)

compile_pip_requirements(
    name = "requirements",
    extra_args = ["--allow-unsafe"],
    requirements_in = "requirements.in",
    requirements_txt = "requirements.txt",
)
