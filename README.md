# Gematria - machine learning for machine code

Contains sources of Gematria, a framework for machine learning on machine code.
It includes implementations of the
[GRANITE model](https://arxiv.org/abs/2210.03894) and the
[Ithemal hierarchical LSTM model](https://arxiv.org/abs/1808.07412) for learning
inverse throughput of basic blocks.

## Installation

### Requirements and installation

Our models are built on top of TensorFlow 2.x (using the TensorFlow 1.x
compatibility layer) in a mix of C++ and Python. Most of the training code is
written in Python; we use C++ for the more demanding parts of the code like
graph construction. We use [pybind11](https://github.com/pybind/pybind11) to
make C++ APIs available in Python.

Basic requirements that need to be installed before starting:

*   [Bazel 6.0](https://bazel.build) or newer.
*   A C++ compiler supported by Bazel that compiles C++17. Recent versions of
    GCC and Clang on Linux both fit the bill.
*   Python 3.10 or newer.
*   Git.
*   [PIP](https://pypi.org/project/pip/).

Additional dependencies, including TensorFlow, Protocol buffers, and different
Python libraries are installed through PIP and through Bazel's `WORKSPACE` file.
We strongly recommend using
[virtualenv](https://pypi.org/project/virtualenv/) to install Python packages to
avoid dependency version conflicts with other libraries.

```shell
# Get the source code.
$ git clone https://github.com/google/gematria.git
$ cd gematria

# Set up virtualenv.
$ pip install virtualenv
$ virtualenv env
$ . env/bin/activate

# Install Python dependencies.
$ pip install -r requirements.in

# Build the project, run tests, ...
$ bazel build ...
$ bazel test ...
```

#### Building with CMake

A subset of the project, consisting of tools and libraries we eventually plan to
merge in the LLVM monorepo, are built with cmake. The requirements are inherited
from [LLVM](https://llvm.org/docs/GettingStarted.html#requirements), as we use
LLVM's "external project" mechanism to build.

First, build TFLite. In addition to the requirements above, see also
[these prerequisites](https://github.com/google/ml-compiler-opt#prerequisites),
noting the reference to the buildbot script which lists additional packages.

Then:

<!--* pragma: { seclinter_this_is_fine: true } *-->
```shell
mkdir /tmp/tflite && cd /tmp/tflite
curl https://raw.githubusercontent.com/google/ml-compiler-opt/main/buildbot/build_tflite.sh | bash
```
<!--* pragma: { seclinter_this_is_fine: false } *-->

This should produce a `/tmp/tflite/tflite.cmake`.

```shell
cd ${GEMATRIA_SRC}
mkdir cmake-build && cd cmake-build
cmake -GNinja -DCMAKE_BUILD_TYPE=Release \
  -C /tmp/tflite/tflite.cmake \
  ${LLVM_PROJECT_SRC}/llvm \
  -DLLVM_EXTERNAL_PROJECTS=gematria \
  -DLLVM_EXTERNAL_GEMATRIA_SOURCE_DIR=${GEMATRIA_SRC}
ninja llvm-granite llvm-cm
```

Where `LLVM_PROJECT_SRC` is the absolute path to your local llvm repo, and
`GEMATRIA_SRC` the path to this (the gematria) repo.

To run the `llvm-cm` tests, you can run the following target:

```shell
ninja check-llvm-tools-llvm-cm
```

### Platform Support

We develop and test our code on Linux and x86-64, and we test it on Mac OS X and
ARM. While we did not test it, we expect it to work with minimal changes also on
other architectures and platforms that run TensorFlow.

## Using the models

See the [training](g3doc/training.md) guide and guides for [Python inference](g3doc/inference-api.md) and [C++ inference](g3doc/granite-inference-api.md).

## Repository structure

See the [separate document](g3doc/code-structure.md).

## Get Involved

*   Issue tracker: https://github.com/google/Gematria/issues

We welcome patches -- see [CONTRIBUTING](CONTRIBUTING) for more information on
how to submit a patch.

## Cite us

```
@inproceedings{granite:iiswc:2022,
  author = {O. Sykora and P. Phothilimthana and C. Mendis and A. Yazdanbakhsh},
  booktitle = {2022 IEEE International Symposium on Workload Characterization (IISWC)},
  title = {{GRANITE: A Graph Neural Network Model for Basic Block Throughput Estimation}},
  year = {2022},
}
```
