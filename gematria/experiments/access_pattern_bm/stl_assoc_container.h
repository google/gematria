#ifndef ACCESS_PATTERN_BM_STL_ASSOC_CONTAINER_H_
#define ACCESS_PATTERN_BM_STL_ASSOC_CONTAINER_H_

#include <immintrin.h>

#include <random>

namespace gematria {

static std::default_random_engine generator;
static std::uniform_int_distribution<int> distribution(0, 1023);

template <typename Container>
Container *CreateRandomSTLAssocContainer(std::size_t size) {
  static_assert(std::is_same<typename Container::key_type, int>::value,
                "Container must have `int` keys.");
  static_assert(std::is_same<typename Container::mapped_type, int>::value,
                "Container must have `int` mapped values.");

  auto container = new Container;

  for (int i = 0; i < size; ++i) {
    container->insert({i, distribution(generator)});
  }

  return container;
}

template <typename Container>
void FlushSTLAssocContainerFromCache(Container *container) {
  _mm_mfence();
  for (typename Container::iterator it = container->begin();
       it != container->end(); ++it) {
    _mm_clflushopt(&it->first);
    _mm_clflushopt(&it->second);
  }
  _mm_mfence();
}

template <typename Container>
void DeleteSTLAssocContainer(Container *container) {
  delete container;
}

}  // namespace gematria

#endif
