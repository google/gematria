#ifndef ACCESS_PATTERN_BM_STL_CONTAINER_H_
#define ACCESS_PATTERN_BM_STL_CONTAINER_H_

#include <immintrin.h>

#include <random>
#include <vector>

static std::default_random_engine generator;
static std::uniform_int_distribution<int> distribution(0, 1023);

template <typename Container>
Container *CreateRandomSTLContainer(std::size_t size) {
  static_assert(std::is_same<typename Container::value_type, int>::value,
                "Container must hold `int`s.");

  auto elements = std::vector<int>(size);
  for (int &element : elements) element = distribution(generator);
  auto container = new Container(elements.begin(), elements.end());

  return container;
}

template <typename Container>
void FlushSTLContainerFromCache(Container *container) {
  _mm_mfence();
  for (auto &element : *container) {
    _mm_clflushopt(&element);
  }
  _mm_mfence();
}

template <typename Container>
void DeleteSTLContainer(Container *container) {
  delete container;
}

#endif