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

#include "gematria/experiments/access_pattern_bm/vec_of_vec_matrix.h"

#include <vector>

#include "benchmark/benchmark.h"

namespace gematria {
namespace {

void BM_FlushVecOfVecMatrixFromCache(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random matrix.
  auto matrix = CreateRandomVecOfVecMatrix(size);

  for (auto _ : state) {
    FlushVecOfVecMatrixFromCache(matrix);
  }
}

BENCHMARK(BM_FlushVecOfVecMatrixFromCache)->Range(1 << 4, 1 << 12);

void BM_VecOfVecMatrix_NoFlush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random matrix.
  auto matrix = CreateRandomVecOfVecMatrix(size);
  // auto mock = CreateRandomVecOfVecMatrix(size);

  int sum = 0;
  for (auto _ : state) {
    // FlushVecOfVecMatrixFromCache(mock);

    // Loop over the matrix, doing some dummy
    // operations along the way.
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        sum += (*matrix)[i][j];
      }
    }

    benchmark::DoNotOptimize(sum);
    sum = 0;
  }
}

BENCHMARK(BM_VecOfVecMatrix_NoFlush)->Range(1 << 4, 1 << 12);

void BM_VecOfVecMatrix_Flush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random matrix.
  auto matrix = CreateRandomVecOfVecMatrix(size);

  int sum = 0;
  for (auto _ : state) {
    FlushVecOfVecMatrixFromCache(matrix);

    // Loop over the matrix, doing some dummy
    // operations along the way.
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        sum += (*matrix)[i][j];
      }
    }

    benchmark::DoNotOptimize(sum);
    sum = 0;
  }
}

BENCHMARK(BM_VecOfVecMatrix_Flush)->Range(1 << 4, 1 << 12);

}  // namespace
}  // namespace gematria
