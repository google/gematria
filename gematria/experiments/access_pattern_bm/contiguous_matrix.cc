#include <immintrin.h>

#include <random>

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

void FlushContiguousMatrixFromCache(int *matrix, std::size_t size) {
  constexpr int line_size = 64;  // For all modern Intel x86 processors (?)
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
