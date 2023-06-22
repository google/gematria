#include <benchmark/benchmark.h>

#include <map>
#include <unordered_map>

#include "stl_assoc_container.h"

template <typename Container>
static void BM_FlushSTLAssocContainerFromCache(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random associative container
  auto container = CreateRandomSTLAssocContainer<Container>(size);

  for (auto _ : state) {
    FlushSTLAssocContainerFromCache(container);
  }

  // Deallocate memory associated with the associative container
  DeleteSTLAssocContainer(container);
}

BENCHMARK_TEMPLATE(BM_FlushSTLAssocContainerFromCache, std::map<int, int>)
    ->Range(1 << 4, 1 << 16);
BENCHMARK_TEMPLATE(BM_FlushSTLAssocContainerFromCache,
                   std::unordered_map<int, int>)
    ->Range(1 << 4, 1 << 16);

template <typename Container>
static void BM_STLAssocContainer_NoFlush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random associative container
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

  // Deallocate memory associated with the associative container
  DeleteSTLAssocContainer(container);
  // DeleteSTLAssocContainer(mock);
}

BENCHMARK_TEMPLATE(BM_STLAssocContainer_NoFlush, std::map<int, int>)
    ->Range(1 << 4, 1 << 16);
BENCHMARK_TEMPLATE(BM_STLAssocContainer_NoFlush, std::unordered_map<int, int>)
    ->Range(1 << 4, 1 << 16);

template <typename Container>
static void BM_STLAssocContainer_Flush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random associative container
  auto container = CreateRandomSTLAssocContainer<Container>(size);

  int sum = 0;
  for (auto _ : state) {
    FlushSTLAssocContainerFromCache(container);

    // Loop over the associative container, doing some dummy
    // operations along the way.
    for (typename Container::iterator it = container->begin();
         it != container->end(); ++it) {
      sum += it->second;
    }

    benchmark::DoNotOptimize(sum);
    sum = 0;
  }

  // Deallocate memory associated with the associative container
  DeleteSTLAssocContainer(container);
}

BENCHMARK_TEMPLATE(BM_STLAssocContainer_Flush, std::map<int, int>)
    ->Range(1 << 4, 1 << 16);
BENCHMARK_TEMPLATE(BM_STLAssocContainer_Flush, std::unordered_map<int, int>)
    ->Range(1 << 4, 1 << 16);

BENCHMARK_MAIN();
