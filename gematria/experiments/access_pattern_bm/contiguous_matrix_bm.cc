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

#include <iostream>

#include "benchmark/benchmark.h"
#include "gematria/experiments/access_pattern_bm/contiguous_matrix.h"

namespace gematria {
namespace {

void BM_FlushContiguousMatrixFromCache(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random matrix.
  auto matrix = CreateRandomContiguousMatrix(size);

  for (auto _ : state) {
    FlushContiguousMatrixFromCache(matrix, size);
  }
}

BENCHMARK(BM_FlushContiguousMatrixFromCache)->Range(1 << 4, 1 << 12);

void BM_ContiguousMatrix_NoFlush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random matrix.
  auto matrix = CreateRandomContiguousMatrix(size);
  // auto mock = CreateRandomContiguousMatrix(size);

  int sum = 0;
  for (auto _ : state) {
    // FlushContiguousMatrixFromCache(mock, size);

    // Loop over the matrix, doing some dummy operations along the way.
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        sum += matrix[size * i + j];
      }
    }

    benchmark::DoNotOptimize(sum);
    sum = 0;
  }
}

BENCHMARK(BM_ContiguousMatrix_NoFlush)->Range(1 << 4, 1 << 12);

void BM_ContiguousMatrix_Flush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random matrix.
  auto matrix = CreateRandomContiguousMatrix(size);

  int sum = 0;
  for (auto _ : state) {
    FlushContiguousMatrixFromCache(matrix, size);

    // Loop over the matrix, doing some dummy operations along the way.
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        sum += matrix[size * i + j];
      }
    }

    benchmark::DoNotOptimize(sum);
    sum = 0;
  }
}

BENCHMARK(BM_ContiguousMatrix_Flush)->Range(1 << 4, 1 << 12);

}  // namespace
}  // namespace gematria
