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

#include <immintrin.h>

#include <memory>
#include <random>
#include <vector>

namespace gematria {

static std::default_random_engine generator;
static std::uniform_int_distribution<int> distribution(0, 1023);

std::unique_ptr<std::vector<std::vector<int>>> CreateRandomVecOfVecMatrix(
    const std::size_t size) {
  auto matrix = std::make_unique<std::vector<std::vector<int>>>(size, std::vector<int>(size));

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      (*matrix)[i][j] = distribution(generator);
    }
  }

  return matrix;
}

void FlushVecOfVecMatrixFromCache(std::unique_ptr<std::vector<std::vector<int>>> &matrix) {
  const std::size_t size = matrix->size();
  constexpr int line_size = 64;

  _mm_mfence();
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; j += line_size / sizeof(int)) {
      _mm_clflushopt(&(*matrix)[i][j]);
    }
  }
  _mm_mfence();
}

}  // namespace gematria
