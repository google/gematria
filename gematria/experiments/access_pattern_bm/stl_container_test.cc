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

#include "gematria/experiments/access_pattern_bm/stl_container.h"

#include <deque>
#include <list>
#include <memory>
#include <set>

#include "benchmark/benchmark.h"
#include "gematria/experiments/access_pattern_bm/configuration.h"

namespace gematria {
namespace {

template <typename Container>
void BM_FlushSTLContainerFromCache(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random container.
  auto container = CreateRandomSTLContainer<Container>(size);

  for (auto _ : state) {
    FlushSTLContainerFromCache(container.get());
  }
}

BENCHMARK(BM_FlushSTLContainerFromCache<std::multiset<int>>)
    ->Range(1 << 4, 1 << 16);
BENCHMARK(BM_FlushSTLContainerFromCache<std::list<int>>)
    ->Range(1 << 4, 1 << 16);
BENCHMARK(BM_FlushSTLContainerFromCache<std::deque<int>>)
    ->Range(1 << 4, 1 << 16);

template <typename Container>
void BM_STLContainer_NoFlush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random container.
  auto container = CreateRandomSTLContainer<Container>(size);
  std::unique_ptr<Container> mock;
  if (kBalanceFlushingTime) {
    mock = CreateRandomSTLContainer<Container>(size);
  }

  for (auto _ : state) {
    int sum = 0;
    if (kBalanceFlushingTime) {
      FlushSTLContainerFromCache(mock.get());
    }

    // Loop over the container, doing some dummy
    // operations along the way.
    for (auto element : *container) {
      sum += element;
    }

    benchmark::DoNotOptimize(sum);
  }
}

BENCHMARK(BM_STLContainer_NoFlush<std::multiset<int>>)->Range(1 << 4, 1 << 16);
BENCHMARK(BM_STLContainer_NoFlush<std::list<int>>)->Range(1 << 4, 1 << 16);
BENCHMARK(BM_STLContainer_NoFlush<std::deque<int>>)->Range(1 << 4, 1 << 16);

template <typename Container>
void BM_STLContainer_Flush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random container.
  auto container = CreateRandomSTLContainer<Container>(size);

  for (auto _ : state) {
    int sum = 0;
    FlushSTLContainerFromCache(container.get());

    // Loop over the container, doing some dummy
    // operations along the way.
    for (auto element : *container) {
      sum += element;
    }

    benchmark::DoNotOptimize(sum);
    sum = 0;
  }
}

BENCHMARK(BM_STLContainer_Flush<std::multiset<int>>)->Range(1 << 4, 1 << 16);
BENCHMARK(BM_STLContainer_Flush<std::list<int>>)->Range(1 << 4, 1 << 16);
BENCHMARK(BM_STLContainer_Flush<std::deque<int>>)->Range(1 << 4, 1 << 16);

}  // namespace
}  // namespace gematria
