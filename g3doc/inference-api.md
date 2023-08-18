# Gematria inference API

This document describes the APIs for inference with trained Gematria models.

## Command-line inference API

The module
[`gematria.model.python.main_function`](http://gematria/model/python/main_function.py)
provides an inference mode where the binary reads a
[`.tfrecord` file](representation.md) where each record contains a single
`BasicBlockWithThroughputProto` in the serialized proto format. The output is
written in the same format and preserving the order of the samples to another
file.

Model binaries using this module support inference automatically. The required
flags to run inference are:

-   `--gematria_action=predict`: required to run the model in batch inference
    mode.
-   `--gematria_input_file={filename}`: The path to the input `.tfrecord` file.
-   `--gematria_output_file={filename}`: The path to the output `.tfrecord`
    file.
-   `--gematria_checkpoint_file={checkpoint}`: The path to a TensorFlow
    checkpoint that contains the trained model used for inference.

In addition to these flags, you must also provide the parameters of the model in
model-specific flags with the same values as those used to train the model.

Example command-line:

```shell
$ bazel run -c opt \
    //gematria/granite/python:run_granite_model \
    -- \
    --gematria_action=predict \
    --gematria_input_file=/tmp/input.tfrecord \
    --gematria_output_file=/tmp/output.tfrecord \
    --gematria_tokens_file=/tmp/tokens.txt \
    --gematria_checkpoint_file=/tmp/granite_model/model.ckpt-10000
```

## Python inference API

Python code can interact directly with the Gematria model class, without going
through a `.tfrecord` file or. Gematria models based on the
`gematria.model.python.main_function.ModelBase` class all provide a `Predict`
method that takes a list of `BasicBlockWithThroughputProto` and returns a list
of the same protos with the predictions added to them.

Example code using the Python API:

```python
import tensorflow.compat.v1 as tf

from gematria.basic_block.python import tokens
from gematria.granite.python import token_graph_builder_model
from gematria.model.python import options

_INPUT_BLOCKS = []     # Replace with a list of BasicBlockWithThroughputProtos.
_CHECKPOINT_FILE = ''  # Replace with a path to the TensorFlow checkpoint.

_MODEL_TOKENS = []     # Replace with a list of tokens used for training the model.

model = token_graph_builder_model.TokenGraphBuilderModel(
    tokens=_MODEL_TOKENS,
    dtype=tf.dtypes.float32,
    immediate_token=tokens.IMMEDIATE,
    fp_immediate_token=tokens.IMMEDIATE,
    address_token=tokens.ADDRESS,
    memory_token=tokens.MEMORY,
    node_embedding_size=256,
    edge_embedding_size=256,
    global_embedding_size=256,
    node_update_layers=(256, 256),
    edge_update_layers=(256, 256),
    global_update_layers=(256, 256),
    readout_layers=(256, 256),
    task_readout_layers=(256, 256),
    num_message_passing_iterations=8,
    loss_type=options.LossType.MEAN_SQUARED_ERROR,
    loss_normalization=options.ErrorNormalization.PERCENTAGE_ERROR
)
model.Initialize()
with tf.Session() as sess:
  saver = tf.train.Saver()
  saver.restore(sess, _CHECKPOINT_FILE)
  output_blocks = model.Predict(sess, _INPUT_BLOCKS)
```

## In-process C++ inference API

For models based on graph neural network, we provide a native C++ API based on
the TensorFlow Lite Framework. This API is described in a
[separate document](granite-inference-api.md).
