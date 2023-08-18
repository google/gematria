# Granite C++ inference API

This document describes the "native" C++ inference API available for Gematria
models based on the
[TokenGraphBuilderModel](../gematria/granite/python/token_graph_builder_model.py)
class. This is provided in addition to the
[Python and command-line APIs](inference-api.md) available to all Gematria
models.

## Using the C++ API

To use the C++ API, you need to add the
[inference library](../gematria/granite/graph_builder_model_inference.h) as a
dependency. You will also need one or more `.tflite` file with a trained GRANITE
model, or bundle it with the binary in some other way.

The API has a single class `GraphBuilderModelInference`. The general workflow
when using the API is:

1.  Load the TensorFlow Lite model as a
    [`FlatBufferModel`](https://www.tensorflow.org/lite/api_docs/cc/namespace/tflite)
    object using the factory methods of `tflite::FlatBufferModel`.
2.  Create a `GraphBuilderModelInference` object using the TensorFlow Lite
    model. The model contains all the necessary parameters to create the
    inference object.
3.  For each batch, add basic blocks to the batch using
    `GraphBuilderModelInference::AddBasicBlockToBatch()`, then compute the
    predictions for the basic blocks in the batch using
    `GraphBuilderModelInference::RunInference()`.
4.  To process more batches: call `GraphBuilderModelInference::Reset()` and
    repeat from step 3.

The C++ API uses the [TensorFlow Lite](https://www.tensorflow.org/lite/)
framework. It wraps the TensorFlow Lite classes and functions and provides a
convenient API based on the
[`BasicBlock`](../gematria/basic_block/basic_block.h) data structures used by
the rest of Gematria code.

By default, the Gematria inference library links just the necessary TensorFlow
Lite libraries. You may have to add additional dependencies to use GPU
processing when available.

## Exporting models to the .tflite format

A `.tflite` file contains a TensorFlow Lite computation graph, and the files are
typically created from a trained TensorFlow model, i.e. a TensorFlow graph and
variable values.

The conversion of Gematria models to the `.tflite` format is typically done in
two steps:

1.  Export a frozen
    [`tensorflow.GraphDef`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/GraphDef)
    from a model and a checkpoint with weights. This can be done by passing
    `--gematria_action=export_graph_def` and `--gematria_checkpoint_file` to the
    model binary. The command-line flags specifying the hyper-parameters of the
    model must be set the same way as they were set during the training of the
    model.

    With a checkpoint file specified, the export will replace all Variable nodes
    in the TensorFlow graph with constant nodes containing the values of the
    variables from the checkpoint.

    Example:

    ```bash
    bazel run -c opt \
      //gematria/granite:token_graph_builder_model_main \
      -- \
      --logtostderr \
      --gematria_action=export_graph_def \
      --gematria_edge_embedding_size=256 \
      --gematria_edge_update_layers=256,256 \
      --gematria_global_embedding_size=256 \
      --gematria_global_update_layers=256,256 \
      --gematria_graph_module_layer_normalization \
      --gematria_graph_module_residual_connections \
      --gematria_node_embedding_size=256 \
      --gematria_node_update_layers=256,256 \
      --gematria_num_message_passing_iterations=4 \
      --nogematria_readout_input_layer_normalization \
      --gematria_readout_layers='' \
      --gematria_readout_residual_connections \
      --gematria_seq2seq \
      --gematria_task_names=ivb \
      --gematria_task_names=hsw \
      --gematria_task_names=skx \
      --nogematria_task_readout_input_layer_normalization \
      --gematria_task_readout_layers= \
      --gematria_task_readout_residual_connections \
      --gematria_use_sent_edges \
      --nogematria_use_seq2seq_loss \
      --gematria_checkpoint_file=/tmp/model.ckpt-1 \
      --gematria_graph_def_file=/tmp/gnn_frozen_graph.pbtxt
    ```

2.  Convert the model from the `GraphDef` format to the `.tflite` format. This
    is done using the
    [`tflite_convert`](https://www.tensorflow.org/lite/models/convert/convert_models#command_line_tool_)
    tool. For convenience and to ensure that models are exported with the inputs
    and outputs always in the same order, we provide a
    [conversion script](../gematria/granite/convert_gb_token_model_to_tflite.sh).

    Example:

    ```bash
    gematria/granite/convert_gb_token_model_to_tflite.sh \
      --gematria_input_graphdef /tmp/gnn_frozen_graph.pbtxt \
      --gematria_output_tflite /tmp/gnn.tflite
    ```
