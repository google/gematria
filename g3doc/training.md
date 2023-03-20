# Training Gematria models

This document describes how to train Gematria models.

## Command-line training API

The module
[`gematria.model.python.main_function`](http://gematria/model/python/main_function.py)
provides a training mode where the binary reads training data from a
[`.tfrecord` file](representation.md) where each record contains a single
`BasicBlockWithThroughputProto` in the serialized proto format, and can be
configured to write TensorFlow 1.x checkpoints that can be loaded by the models
for inference.

Model binaries using this module support inference automatically. The required
flags to run inference are:

-   `--gematria_action=train`: required to run the model in batch inference mode
    (`train` is the default value).
-   `--gematria_input_file={filename}`: The path to the `.tfrecord` file(s) with
    training data.
-   `--gematria_checkpoint_dir={checkpoint_dir}`: The path to a directory where
    checkpoint files will be stored. When training is interrupted, it resumes
    from the last checkpoint.
-   `--gematria_training_num_epochs={num_batches}`: The number of training steps
    to take during the training. When zero or negative, the training runs
    indefinitely and must be manually interrupted.
-   `--gematria_max_blocks_in_batch={num_blocks}`: The maximal number of basic
    blocks per batch. By default, the whole data set is used.
-   `--gematria_throughput_source_filter`: A regexp matching throughput sources
    in the `BasicBlockWithThroughputProto`s in the training data that are used
    for training

[`main_function.py`](http://gematria/model/python/main_function.py) defines many
more flags that can be used to tune the training behavior.

In addition to model-independent flags, you must also provide the parameters of
the model in model-specific flags with the same values as those used to train
the model.

Example command-line:

```shell
$ bazel run -c opt \
    //gematria/sequence/python:run_sequence_model \
    -- \
    --gematria_action=train \
    --gematria_tokens_file /tmp/tokens.txt \
    --gematria_input_file /tmp/blocks.tfrecord \
    --gematria_throughput_source_filter="ithemal: kind=KIND_MEASURED, uarch=ARCH_HASWELL" \
    --gematria_training_num_epochs=0 \
    --gematria_learning_rate=0.01 \
    --gematria_loss_type=MEAN_ABSOLUTE_ERROR \
    --gematria_loss_normalization=PERCENTAGE_ERROR \
    --gematria_max_blocks_in_batch=100 \
    --gematria_checkpoint_dir=/tmp/hlstm_model
```
