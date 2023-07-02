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

#include <random>

#include "absl/base/optimization.h"

namespace gematria {

static std::default_random_engine generator;
static std::uniform_int_distribution<int> distribution(0, 1023);

int *CreateRandomContiguousMatrix(const std::size_t size) {
  auto matrix = new int[size * size];

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      matrix[size * i + j] = distribution(generator);
    }
  }

  return matrix;
}

void FlushContiguousMatrixFromCache(int *matrix, const std::size_t size) {
  constexpr int line_size = ABSL_CACHELINE_SIZE;
  const char *ptr = (const char *)matrix;
  const char *end = (const char *)(matrix + (size + 1) * sizeof(int));

  _mm_mfence();
  while (ptr <= end) {
    _mm_clflushopt(ptr);
    ptr += line_size;
  }
  _mm_mfence();
}

void DeleteContiguousMatrix(int *matrix) { delete[] matrix; }

}  // namespace gematria
