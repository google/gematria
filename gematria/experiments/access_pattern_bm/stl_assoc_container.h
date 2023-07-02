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

#ifndef GEMATRIA_EXPERIMENTS_ACCESS_PATTERN_BM_STL_ASSOC_CONTAINER_H_
#define GEMATRIA_EXPERIMENTS_ACCESS_PATTERN_BM_STL_ASSOC_CONTAINER_H_

#include <immintrin.h>

#include <random>

namespace gematria {

static std::default_random_engine generator;
static std::uniform_int_distribution<int> distribution(0, 1023);

template <typename Container>
Container *CreateRandomSTLAssocContainer(const std::size_t size) {
  static_assert(std::is_same<typename Container::key_type, int>::value,
                "Container must have `int` keys.");
  static_assert(std::is_same<typename Container::mapped_type, int>::value,
                "Container must have `int` mapped values.");

  auto container = new Container;

  for (int i = 0; i < size; ++i) {
    container->emplace(i, distribution(generator));
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
