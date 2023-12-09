from gematria.basic_block.python import basic_block
from gematria.basic_block.python import basic_block_protos
from gematria.proto import basic_block_pb2
from gematria.proto import throughput_pb2
from gematria.proto import canonicalized_instruction_pb2
from gematria.io.python import tfrecord

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging

import numpy as np
_CanonicalizedInstructionProto = (
    canonicalized_instruction_pb2.CanonicalizedInstructionProto
)

r"""Generates tokens from a Gematria data set.


Usage:
  gen_tokens \
      --gematria_input_tfrecord=/tmp/bhive/skl.tfrecord \
      --gematria_output_tokens=/tmp/bhive/skl_tokens.txt \

"""

_INPUT_TFRECORD_FILE = flags.DEFINE_string(
    'gematria_input_tfrecord',
    None,
    'The name of the TFRecord file to read the tokens from.',
    required=True,
)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    output_blocks = list(
        tfrecord.read_protos((_INPUT_TFRECORD_FILE.value,), throughput_pb2.BasicBlockWithThroughputProto)
    )
    y_actual = []  
    y_predict = [] 
    for block in output_blocks:
        print(block)
        assert(len(block.inverse_throughputs) == 2)
        y_actual.append(block.inverse_throughputs[0].inverse_throughput_cycles[0])
        y_predict.append(block.inverse_throughputs[1].inverse_throughput_cycles[0])

    print("y true is:")
    print(y_actual)
    print("y pred is:")
    print(y_predict)
    y_actual = np.array(y_actual)
    y_predict = np.array(y_predict)

  # Ensure that y_actual and y_predict have the same length
    if y_actual.shape != y_predict.shape:
        raise ValueError("The shapes of y_actual and y_predict must be the same")
    # Find the 10th and 90th percentiles of y_actual
    p10 = np.percentile(y_actual, 10)
    p90 = np.percentile(y_actual, 90)
    # Filter both arrays to ignore the bottom and top 10 percentiles
    filtered_indices = (y_actual >= p10) & (y_actual <= p90)
    filtered_y_actual = y_actual[filtered_indices]
    filtered_y_predict = y_predict[filtered_indices]
    # Compute MAE for the filtered data
    mae = np.mean(np.abs(filtered_y_actual - filtered_y_predict))
    # Compute MSE for the filtered data
    mse = np.mean((filtered_y_actual - filtered_y_predict) ** 2)
    print(f"Mean Absolute Error (MAE) for the 10th to 90th percentile range: {mae}")
    print(f"Mean Absolute Error (MSE) for the 10th to 90th percentile range: {mse}")

      
  

if __name__ == '__main__':
  app.run(main)