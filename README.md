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
*   A C++ compiler supported by Bazel that compiles C++20. Recent versions of
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
$ pip install -r requirements.txt

# On OS X only. The dependencies of tensorflow-ranking are not set up correctly
# and it needs to be installed manually.
$ pip install --no-deps tensorflow-ranking.

# Build the project, run tests, ...
$ bazel build ...
$ bazel test ...
```

### Updating Dependencies

If you are adding a new python dependency, make sure to add it along with relevant
version information to the `requirements.in` file. Then regenerated the
`requirements.txt` using Bazel and install the new dependencies with `pip`:

```shell
$ bazel run :requirements.update
$ pip install -r requirements.txt
```

### Platform Support

We develop and test our code on Linux and x86-64, and we test it on Mac OS X and
ARM. While we did not test it, we expect it to work with minimal changes also on
other architectures and platforms that run TensorFlow.

## Using the models

See the [training](g3doc/training.md) and [inference](g3doc/inference-api.md)
guides.

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
