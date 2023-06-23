#include <benchmark/benchmark.h>

#include <vector>

#include "vec_of_vec_matrix.h"

namespace gematria {

static void BM_FlushVecOfVecMatrixFromCache(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random matrix
  auto matrix = CreateRandomVecOfVecMatrix(size);

  for (auto _ : state) {
    FlushVecOfVecMatrixFromCache(matrix);
  }

  // Deallocate memory associated with the matrix
  DeleteVecOfVecMatrix(matrix);
}

BENCHMARK(BM_FlushVecOfVecMatrixFromCache)->Range(1 << 4, 1 << 12);

static void BM_VecOfVecMatrix_NoFlush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random matrix
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

  // Deallocate memory associated with the matrix
  DeleteVecOfVecMatrix(matrix);
  // DeleteVecOfVecMatrix(mock);
}

BENCHMARK(BM_VecOfVecMatrix_NoFlush)->Range(1 << 4, 1 << 12);

static void BM_VecOfVecMatrix_Flush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random matrix
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

  // Deallocate memory associated with the matrix
  DeleteVecOfVecMatrix(matrix);
}

BENCHMARK(BM_VecOfVecMatrix_Flush)->Range(1 << 4, 1 << 12);

}  // namespace gematria

BENCHMARK_MAIN();
