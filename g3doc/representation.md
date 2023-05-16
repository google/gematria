# Gematria data representation

This document provides an overview of the data format used by Gematria models to
represent basic blocks and instructions in them.

## Dataset representation

Most Gematria models and tools, and in particular the training and inference
pipeline, consume the basic block data as `.tfrecord` files that contain one
`BasicBlockWithThroughputProto` that contains the data of the basic block.

The proto has fields for two different representations of the basic block. Which
representation is used for inference and the exact format of the representation
is architecture-, problem- and model-dependent.

The basic blocks passed to the inference APIs as inputs may contain values in
`inverse_throughputs`. These values are preserved unchanged by all models, and
the models add their own predictions as a new value at the end of the
`inverse_throughputs` repeated field.

The representations of instructions in the proto are

1.  `MachineInstructionProto` for raw instructions, and
2.  `CanonicalizedInstructionProto` for canonicalized (assembly-style)
    instructions.

NOTE: As of March 2023, all models consume only the canonicalized instruction
data.

## Basic block representation

Each basic block is stored in a `BasicBlockWithThroughputProto`. The proto has
two main components:

*   `basic_block` contains the instructions of the basic block and other
    information. Most models depend only on the `canonicalized_instructions`
    field; models using a specialized embedder may use other fields.
*   [`inverse_throughputs`](symbol:inverse_throughputs$ class:exegesis.BasicBlockWIthThroughputProto)
    contains the inverse throughput data for the basic block. Each
    `inverse_throughputs` entry for the basic block corresponds to a measurement
    from a particular system, or prediction from a particular model, identified
    by the field `source`. A single entry allows providing multiple values for
    the throughput, e.g. in case of noisy measurements.

    The training pipeline expects the golden labels as throughputs from a given
    source (provided in the command-line flags); the inference pipeline adds a
    new entry with predictions from the model.

## Canonicalized instructions

This approach is inspired by the
[Ithemal paper](https://arxiv.org/abs/1808.07412). Each instruction is
represented as a sequence of tokens, based on the assembly code of the
instruction.

The stream has the structure *prefixes and mnemonic*, *\<D\>*, *input operand
tokens*, *\<D\>*, *output operand tokens*, *\<D\>*, where *\<D\>* is a special
delimiter token. When an instruction has a prefix, then the prefix is a separate
token. Unlike the assembly code, the canonicalized token sequence contains
tokens also for the implicit input and output operands.

| Assembly                   | Canonicalized instruction                       |
| -------------------------- | ----------------------------------------------- |
| `MOV EAX, EBX`             | `["MOV", "_D_", "EAX", "_D_", "EBX", "_D_"]`    |
| `ADD EAX, EBX`             | `["ADD", "_D_", "EAX","EFLAGS", "_D_", "EBX",   |
:                            : "_D_"]`                                         :
| `ADD EAX, DWORD PTR [RBX]` | `["ADD", "_D_", "EAX", "_D_", "EAX",            |
:                            : "_MEMORY_", "_ADDRESS_", "RBX",                 :
:                            : "_NO_REGISTER_", "_D_"]`                        :
| `REP MOVSB`                | `["REP", "MOVSB", "_D_", "_mem_", "RSI", "RDI", |
:                            : "_D_", "_mem_", "RSI", "RDI", "DF"]`            :

Note that while the canonicalized instructions described in the paper are
linear, in the protos, we store them in a structured format that corresponds to
the syntax of the assembly language. The transformation to the linear form is
only done on the fly, during the training and inference.
