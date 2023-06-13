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
"""Contains functions for performance debugging in Gematria code."""

from collections.abc import Callable, Iterator
import contextlib
import time
from typing import Optional

from absl import logging


@contextlib.contextmanager
def scoped(
    name: str,
    num_iterations: int = 1,
    log_function: Optional[Callable[..., None]] = None,
) -> Iterator[None]:
  """Measures the running time of the code in the context and prints it to log.

  Example usage:
    with scoped('Add one'):
      n = n + 1

  Args:
    name: The name of the measured block. This will be printed to the log.
    num_iterations: The number of iterations ran in the block. When greater than
      one, the timer will print also the average time per iteration.
    log_function: A custom function used for logging. When not specified, the
      timing information is printed to absl.logging.info.

  Yields:
    None. Yielding is used only as a way to transfer control to the measured
    code.
  """
  # NOTE(ondrasej): We need to use time.time(), not time.clock() here. On Linux,
  # time.clock() returns the CPU time over all cores. Which is a useless measure
  # for debugging TensorFlow performance, especially if comparing a CPU against
  # a GPU or a TPU.
  start_time = time.time()
  log_function = log_function or logging.info
  yield
  duration = time.time() - start_time
  if num_iterations > 1:
    log_function('%s: %fs, %fs per iteration', name, duration,
                 duration / num_iterations)
  else:
    log_function('%s: %fs', name, duration)
