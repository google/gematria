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

#include <memory>
#include <vector>

#include "benchmark/benchmark.h"
#include "gematria/experiments/access_pattern_bm/configuration.h"

namespace gematria {
namespace {

void BM_FlushVecOfVecMatrixFromCache(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random matrix.
  auto matrix = CreateRandomVecOfVecMatrix(size);

  for (auto _ : state) {
    FlushVecOfVecMatrixFromCache(matrix.get());
  }
}

BENCHMARK(BM_FlushVecOfVecMatrixFromCache)->Range(1 << 4, 1 << 12);

void BM_VecOfVecMatrix_NoFlush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random matrix.
  auto matrix = CreateRandomVecOfVecMatrix(size);
  std::unique_ptr<std::vector<std::vector<int>>> mock;
  if (kBalanceFlushingTime) {
    mock = CreateRandomVecOfVecMatrix(size);
  }

  for (auto _ : state) {
    int sum = 0;
    if (kBalanceFlushingTime) {
      FlushVecOfVecMatrixFromCache(mock.get());
    }

    // Loop over the matrix, doing some dummy
    // operations along the way.
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        sum += (*matrix)[i][j];
      }
    }

    benchmark::DoNotOptimize(sum);
  }
}

BENCHMARK(BM_VecOfVecMatrix_NoFlush)->Range(1 << 4, 1 << 12);

void BM_VecOfVecMatrix_Flush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random matrix.
  auto matrix = CreateRandomVecOfVecMatrix(size);

  for (auto _ : state) {
    int sum = 0;
    FlushVecOfVecMatrixFromCache(matrix.get());

    // Loop over the matrix, doing some dummy
    // operations along the way.
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        sum += (*matrix)[i][j];
      }
    }

    benchmark::DoNotOptimize(sum);
  }
}

BENCHMARK(BM_VecOfVecMatrix_Flush)->Range(1 << 4, 1 << 12);

}  // namespace
}  // namespace gematria
