// Copyright 2023 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gematria/experiments/access_pattern_bm/contiguous_matrix.h"

#include <immintrin.h>

#include <memory>
#include <random>

#include "absl/base/optimization.h"

namespace gematria {

// Creates a size x size matrix, as an array of (size * size) random integers.
std::unique_ptr<int[]> CreateRandomContiguousMatrix(const std::size_t size) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 1023);

  auto matrix = std::make_unique<int[]>(size * size);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      matrix[size * i + j] = distribution(generator);
    }
  }

  return matrix;
}

// Flushes matrices created by `CreateRandomContiguousMatrix`. Takes the matrix
// itself, along with its size (given that it is a size x size square matrix,
// holding size * size ints).
void FlushContiguousMatrixFromCache(const void *matrix,
                                    const std::size_t size) {
  constexpr int line_size = ABSL_CACHELINE_SIZE;
  const char *begin = reinterpret_cast<const char *>(matrix);
  const char *end = begin + size * size * sizeof(int);

  _mm_mfence();
  for (; begin < end; begin += line_size) {
    _mm_clflushopt(begin);
  }
  _mm_mfence();
}

}  // namespace gematria
