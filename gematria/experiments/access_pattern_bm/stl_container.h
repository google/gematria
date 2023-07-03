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

#ifndef GEMATRIA_EXPERIMENTS_ACCESS_PATTERN_BM_STL_CONTAINER_H_
#define GEMATRIA_EXPERIMENTS_ACCESS_PATTERN_BM_STL_CONTAINER_H_

#include <immintrin.h>

#include <memory>
#include <random>
#include <vector>

namespace gematria {

static std::default_random_engine generator;
static std::uniform_int_distribution<int> distribution(0, 1023);

template <typename Container>
std::unique_ptr<Container> CreateRandomSTLContainer(const std::size_t size) {
  static_assert(std::is_same<typename Container::value_type, int>::value,
                "Container must hold `int`s.");

  auto container = std::make_unique<Container>();
  auto inserter = std::inserter(*(container.get()), container.get()->end());
  for (int i = 0; i < size; ++i, ++inserter) {
    *inserter = distribution(generator);
  }

  return container;
}

template <typename Container>
void FlushSTLContainerFromCache(std::unique_ptr<Container> &container) {
  _mm_mfence();
  for (auto &element : *(container.get())) {
    _mm_clflushopt(&element);
  }
  _mm_mfence();
}

}  // namespace gematria

#endif
