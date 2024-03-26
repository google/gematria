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

#include "gematria/experiments/access_pattern_bm/linked_list.h"

#include <memory>

#include "benchmark/benchmark.h"
#include "gematria/experiments/access_pattern_bm/configuration.h"
#include "gtest/gtest.h"

namespace gematria {
namespace {

TEST(LinkedListTest, CreateRandomLinkedList) {
  constexpr int kSize = 3;
  auto list = CreateRandomLinkedList(kSize);
  const auto *head = list.get();
  for (int i = 0; i < kSize; ++i) {
    ASSERT_NE(head, nullptr);
    EXPECT_GE(head->value, 0);
    EXPECT_LE(head->value, kMaxRandomListValue);
    head = head->next;
  }
  ASSERT_EQ(head, nullptr);
}

// Times flushing an entire linked list from cache - not sure how Google
// Benchmark repeating the test changes the results, i.e. what happens when
// clflush cache misses (which seems to happen a ton).
void BM_FlushLinkedListFromCache(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random linked list.
  auto head = CreateRandomLinkedList(size);

  for (auto _ : state) {
    FlushLinkedListFromCache(head.get());
  }
}

BENCHMARK(BM_FlushLinkedListFromCache)->Range(1 << 4, 1 << 20);

// Traverses a linked list, does not try to directly manipulate the cache in
// any way.
void BM_AccessLinkedList_NoFlush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random linked list.
  auto head = CreateRandomLinkedList(size);
  std::unique_ptr<Node, NodeDeleter> mock;
  if (kBalanceFlushingTime) {
    // mock is used to balance out the extra flushing time to make a better
    // comparison from a time perspective.
    mock = CreateRandomLinkedList(size);
  }

  for (auto _ : state) {
    int sum = 0;
    if (kBalanceFlushingTime) {
      state.PauseTiming();
      FlushLinkedListFromCache(mock.get());
      state.ResumeTiming();
    }

    // Traverse the linked list, doing some arbitrary operations on each element
    // to mimic realistic use to some extent.
    Node *current = head.get();

    while (current) {
      sum += current->value;
      current = current->next;
    }

    benchmark::DoNotOptimize(sum);
  }
}

BENCHMARK(BM_AccessLinkedList_NoFlush)->Range(1 << 4, 1 << 20);

// Traverses a linked list after making sure it is not in the caches by
// explicitly flushing it before the traversal. Again, clflush itself cache
// misses plenty.
void BM_AccessLinkedList_Flush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random linked list.
  auto head = CreateRandomLinkedList(size);

  for (auto _ : state) {
    int sum = 0;
    state.PauseTiming();
    FlushLinkedListFromCache(head.get());
    state.ResumeTiming();

    // Traverse the linked list, doing some arbitrary operations
    // on each element to mimic realistic use to some extent.
    Node *current = head.get();

    while (current) {
      sum += current->value;
      current = current->next;
    }

    benchmark::DoNotOptimize(sum);
  }
}

BENCHMARK(BM_AccessLinkedList_Flush)->Range(1 << 4, 1 << 20);

}  // namespace
}  // namespace gematria
