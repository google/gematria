# Gematria code structure

This document describes the structure of the Gematria repository.

## Basic structure

The Python code is split into modules according to its function; C++ code
follows the same directory structure. By convention, we put all C++ files to
`gematria/{package_name}/`, and the Python libraries under
`gematria/{package_name}/python`.

### Package: basic_block

Contains code and data structures related to in-memory representation of basic
blocks. While we use protos to store the data in rest, most of the code uses a
lightweight data structure that can is easy to share between Python and C++.

### Package: granite

Implementation of the [GRANITE model](https://arxiv.org/abs/2210.03894) and base
classes for building models with Graph neural networks. The graph construction
from basic block data is implemented in C++ for efficiency reasons.

Notable files:

-   `granite/python/run_granite_model.py`: the main module for running the
    GRANITE model.

### Package: io

Input/output utilities.

### Package: model

Contains base classes for building models and the necessary support code for
training and inference (the implementation of the training loop, an inference
loop, and a generic main function for running Gematria models).

Notable files:

-   `model/python/model_base.py`: the base class for all Gematria models.
    Contains most of the model-independent code, like the training and inference
    loops, cost definition, ...
-   `model/python/main_function.py`: contains model-independent code needed to
    launch Gematria models from command-line: definitions of command-line flags,
    and a generic main() function.

### Directory: proto

Protocol buffer definitions used in the project.

### Package: sequence

Base classes and implementation of models that treat the basic block as a
sequence of instructions. In particular, contains implementation of the
[Ithemal model](https://arxiv.org/abs/1808.07412) and the Ithemal+ model
described in the [Granite paper](https://arxiv.org/abs/2210.03894).

Notable files:

-   `sequence/python/run_sequence_model.py`: the main module for running the
    Ithemal and Ithemal+ models.

### Package: testing

Contains helper classes and functions for testing the model and a tiny data set
of basic blocks for testing.

### Package: utils

Contains various utilities that do not fit into other packages.
