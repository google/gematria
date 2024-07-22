# Copyright 2022 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Definitions of enum and option types for Gematria models."""

import enum


# Possible values of the --gematria_action command-line flag.
@enum.unique
class Action(enum.Enum):
  """Specifies the action to take when running a Gematria model.

  Values:
    EVAL: Run continuous evaluation during training.
    EXPORT_GRAPH_DEF: Export a trained model to a file as a frozen graphdef.
    PREDICT: Compute inverse throughput for a collection of basic blocks using
      a trained model.
    TRAIN: Train a model.
  """

  # TODO(ondrasej): See if we can remove VALIDATION, this makes no sense in
  # the open-source world.

  EVAL = 'eval'
  EXPORT_GRAPH_DEF = 'export_graph_def'
  PREDICT = 'predict'
  TRAIN = 'train'


# The list of Action values where input data is not loaded, and we thus do not
# need to check that the list of throughput source filters is set up correctly.
ACTIONS_WITHOUT_INPUT_DATA = (Action.EXPORT_GRAPH_DEF,)


@enum.unique
class LearningRateScheduleType(enum.Enum):
  """The way the learning rate is decayed.

  Values:
    NONE: No learning decay.
    COSINE: Applies cosine decay to the learning rate.
    EXPONENTIAL: Applies exponential decay to the learning rate.
    INVERSE_TIME: Applies inverse time decay to the initial learning rate.
    LINEAR_COSINE: Applies linear cosine decay to the learning rate.
    NATURAL_EXP: Applies natural exponential decay to the initial learning rate.
    NOISY_LINEAR_COSINE: Applies noisy linear cosine decay to the learning rate.
    POLYNOMIAL: Applies a polynomial decay to the learning rate.
  """

  NONE = 1
  COSINE = 2
  EXPONENTIAL = 3
  INVERSE_TIME = 4
  LINEAR_COSINE = 5
  NATURAL_EXP = 6
  NOISY_LINEAR_COSINE = 7
  POLYNOMIAL = 8


@enum.unique
class OptimizerType(enum.Enum):
  """The TF optimizer types used for training.

  Values:
    ADAM: Uses ADAM optimizer for training.
    SGD: Uses GradientDescentOptimizer for training.
    RMSPROP: Uses RMSPropOptimizer for training.
  """

  ADAM = 1
  SGD = 2
  RMSPROP = 3


@enum.unique
class RnnType(enum.Enum):
  """Specifies the type of RNN that is used after GNN.

  Values:
    NONE: No RNN is used, only a GNN.
    LSTM: Long Short-Term Memory layer.
    GRU: Gated Recurrent layer.
  """

  NONE = 0
  LSTM = 1
  GRU = 2


@enum.unique
class LossType(enum.Enum):
  """The types of loss supported in Gematria models.

  Values:
    MEAN_SQUARED_ERROR: The loss is computed as the mean squared error of the
      errors on individual samples (the L2 loss).
    MEAN_ABSOLUTE_ERROR: The loss is computed as the mean of the absolute values
      of the errors of individual samples (the L1 loss).
    HUBER: The loss is the Huber loss; the loss is quadratic around zero, linear
      otherwise. See https://en.wikipedia.org/wiki/Huber_loss.
  """

  MEAN_SQUARED_ERROR = 1
  MEAN_ABSOLUTE_ERROR = 2
  HUBER = 3


@enum.unique
class ErrorNormalization(enum.Enum):
  """The way errors are normalized in loss computation.

  Values:
    NONE: No normalization of errors is applied. The raw difference between the
      expected and the actual value is used.
    PERCENTAGE_ERROR: All errors are normalized by the expected value. This is
      equivalent to computing percentage error.
    EXPECTED_VALUE_GREATER_THAN_ONE: Errors are normalized by the expected value
      when the expected value is greater than one.
  """

  NONE = 1
  PERCENTAGE_ERROR = 2
  EXPECTED_VALUE_GREATER_THAN_ONE = 3


@enum.unique
class EnableFeature(enum.Enum):
  """Controls how features are enabled in the model in relation to a flag.

  The feature flag mode is used to configure insertion points for certain
  features. It specifies whether the feature is always used at the given
  insertion point, it is never used, or its use is controlled by a model-global
  feature flag.
  """

  NEVER = 1
  ALWAYS = 2
  BY_FLAG = 3
