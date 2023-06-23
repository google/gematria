#include <benchmark/benchmark.h>

#include <deque>
#include <list>
#include <set>

#include "stl_container.h"

namespace gematria {

template <typename Container>
static void BM_FlushSTLContainerFromCache(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random container
  auto container = CreateRandomSTLContainer<Container>(size);

  for (auto _ : state) {
    FlushSTLContainerFromCache(container);
  }

  // Deallocate memory associated with the container
  DeleteSTLContainer(container);
}

BENCHMARK_TEMPLATE(BM_FlushSTLContainerFromCache, std::multiset<int>)
    ->Range(1 << 4, 1 << 16);
BENCHMARK_TEMPLATE(BM_FlushSTLContainerFromCache, std::list<int>)
    ->Range(1 << 4, 1 << 16);
BENCHMARK_TEMPLATE(BM_FlushSTLContainerFromCache, std::deque<int>)
    ->Range(1 << 4, 1 << 16);

template <typename Container>
static void BM_STLContainer_NoFlush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random container
  auto container = CreateRandomSTLContainer<Container>(size);
  // auto mock = CreateRandomSTLContainer<Container>(size);

  int sum = 0;
  for (auto _ : state) {
    // FlushSTLContainerFromCache(mock);

    // Loop over the container, doing some dummy
    // operations along the way.
    for (auto element : *container) {
      sum += element;
    }

    benchmark::DoNotOptimize(sum);
    sum = 0;
  }

  // Deallocate memory associated with the container
  DeleteSTLContainer(container);
  // DeleteSTLContainer(mock);
}

BENCHMARK_TEMPLATE(BM_STLContainer_NoFlush, std::multiset<int>)
    ->Range(1 << 4, 1 << 16);
BENCHMARK_TEMPLATE(BM_STLContainer_NoFlush, std::list<int>)
    ->Range(1 << 4, 1 << 16);
BENCHMARK_TEMPLATE(BM_STLContainer_NoFlush, std::deque<int>)
    ->Range(1 << 4, 1 << 16);

template <typename Container>
static void BM_STLContainer_Flush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random container
  auto container = CreateRandomSTLContainer<Container>(size);

  int sum = 0;
  for (auto _ : state) {
    FlushSTLContainerFromCache(container);

    // Loop over the container, doing some dummy
    // operations along the way.
    for (auto element : *container) {
      sum += element;
    }

    benchmark::DoNotOptimize(sum);
    sum = 0;
  }

  // Deallocate memory associated with the container
  DeleteSTLContainer(container);
}

BENCHMARK_TEMPLATE(BM_STLContainer_Flush, std::multiset<int>)
    ->Range(1 << 4, 1 << 16);
BENCHMARK_TEMPLATE(BM_STLContainer_Flush, std::list<int>)
    ->Range(1 << 4, 1 << 16);
BENCHMARK_TEMPLATE(BM_STLContainer_Flush, std::deque<int>)
    ->Range(1 << 4, 1 << 16);

}  // namespace gematria

BENCHMARK_MAIN();
