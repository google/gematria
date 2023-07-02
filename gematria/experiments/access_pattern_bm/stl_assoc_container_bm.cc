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

#include <map>
#include <unordered_map>

#include "benchmark/benchmark.h"
#include "gematria/experiments/access_pattern_bm/stl_assoc_container.h"

namespace gematria {
namespace {

template <typename Container>
void BM_FlushSTLAssocContainerFromCache(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random associative container.
  auto container = CreateRandomSTLAssocContainer<Container>(size);

  for (auto _ : state) {
    FlushSTLAssocContainerFromCache(container);
  }

  // Deallocate memory associated with the associative container.
  DeleteSTLAssocContainer(container);
}

BENCHMARK(BM_FlushSTLAssocContainerFromCache<std::map<int, int>>)
    ->Range(1 << 4, 1 << 16);
BENCHMARK(BM_FlushSTLAssocContainerFromCache<std::unordered_map<int, int>>)
    ->Range(1 << 4, 1 << 16);

template <typename Container>
void BM_STLAssocContainer_NoFlush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random associative container.
  auto container = CreateRandomSTLAssocContainer<Container>(size);
  // auto mock = CreateRandomSTLAssocContainer<Container>(size);

  int sum = 0;
  for (auto _ : state) {
    // FlushSTLAssocContainerFromCache(mock);

    // Loop over the associative container, doing some dummy
    // operations along the way.
    for (typename Container::iterator it = container->begin();
         it != container->end(); ++it) {
      sum += it->second;
    }

    benchmark::DoNotOptimize(sum);
    sum = 0;
  }

  // Deallocate memory associated with the associative container.
  DeleteSTLAssocContainer(container);
  // DeleteSTLAssocContainer(mock);
}

BENCHMARK(BM_STLAssocContainer_NoFlush<std::map<int, int>>)
    ->Range(1 << 4, 1 << 16);
BENCHMARK(BM_STLAssocContainer_NoFlush<std::unordered_map<int, int>>)
    ->Range(1 << 4, 1 << 16);

template <typename Container>
void BM_STLAssocContainer_Flush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random associative container.
  auto container = CreateRandomSTLAssocContainer<Container>(size);

  int sum = 0;
  for (auto _ : state) {
    FlushSTLAssocContainerFromCache(container);

    // Loop over the associative container, doing some dummy
    // operations along the way.
    for (const auto& [key, value] : *container) {
      sum += value;
    }

    benchmark::DoNotOptimize(sum);
    sum = 0;
  }

  // Deallocate memory associated with the associative container.
  DeleteSTLAssocContainer(container);
}

BENCHMARK(BM_STLAssocContainer_Flush<std::map<int, int>>)
    ->Range(1 << 4, 1 << 16);
BENCHMARK(BM_STLAssocContainer_Flush<std::unordered_map<int, int>>)
    ->Range(1 << 4, 1 << 16);

}  // namespace
}  // namespace gematria
