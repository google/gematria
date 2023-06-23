#include <benchmark/benchmark.h>

#include <iostream>

#include "linked_list.h"

namespace gematria {

// Times flushing an entire linked list from cache - not sure how
// Google Benchmark repeating the test changes the results, i.e. what
// happens when clflush cache misses (which seems to happen a ton).
static void BM_FlushLinkedListFromCache(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random linked list
  Node *head = CreateRandomLinkedList(size);

  for (auto _ : state) {
    FlushLinkedListFromCache(head);
  }

  // Free up memory allocated to the linked list
  DeleteLinkedList(head);
}

BENCHMARK(BM_FlushLinkedListFromCache)->Range(1 << 4, 1 << 20);

// Traverses a linked list, does not try to directly manipulate the
// the cache in any way.
static void BM_AccessLinkedList_NoFlush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random linked list
  Node *head = CreateRandomLinkedList(size);
  // Node *mock = CreateRandomLinkedList(size);  <-- mock is
  //                                                 used to balance out
  //                                                 the extra flushing
  //                                                 time to make for a
  //                                                 better
  //                                                 comparison
  //                                                 from
  //                                                 time
  //                                                 perspective

  for (auto _ : state) {
    // Traverse the linked list, doing some arbitrary operations
    // on each element to mimic realistic use to some extent.
    Node *current = head;
    int sum = 0;

    // FlushLinkedListFromCache(mock);           <--

    while (current) {
      sum += current->value;
      current = current->next;
    }

    benchmark::DoNotOptimize(sum);
    sum = 0;
  }

  // Free up memory allocated to the linked list
  DeleteLinkedList(head);
  // DeleteLinkedList(mock);                     <--
}

BENCHMARK(BM_AccessLinkedList_NoFlush)->Range(1 << 4, 1 << 20);

// Traverses a linked list after making sure it is not in the caches
// by explicitly flushing it before the traversal. Again, clflush itself
// cache misses plenty.
static void BM_AccessLinkedList_Flush(benchmark::State &state) {
  const std::size_t size = state.range(0);

  // Create a random linked list
  Node *head = CreateRandomLinkedList(size);

  for (auto _ : state) {
    // Traverse the linked list, doing some arbitrary operations
    // on each element to mimic realistic use to some extent.
    Node *current = head;
    int sum = 0;

    FlushLinkedListFromCache(head);

    while (current) {
      sum += current->value;
      current = current->next;
    }

    benchmark::DoNotOptimize(sum);
    sum = 0;
  }

  // Free up memory allocated to the linked list
  DeleteLinkedList(head);
}

BENCHMARK(BM_AccessLinkedList_Flush)->Range(1 << 4, 1 << 20);

}  // namespace gematria

BENCHMARK_MAIN();
