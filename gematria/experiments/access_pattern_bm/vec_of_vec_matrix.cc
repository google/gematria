#include <immintrin.h>

#include <random>
#include <vector>

static std::default_random_engine generator;
static std::uniform_int_distribution<int> distribution(0, 1023);

std::vector<std::vector<int>> *CreateRandomVecOfVecMatrix(
    const std::size_t size) {
  auto matrix = new std::vector<std::vector<int>>(size, std::vector<int>(size));

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      (*matrix)[i][j] = distribution(generator);
    }
  }

  return matrix;
}

void FlushVecOfVecMatrixFromCache(std::vector<std::vector<int>> *matrix) {
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

void DeleteVecOfVecMatrix(std::vector<std::vector<int>> *matrix) {
  delete matrix;
}