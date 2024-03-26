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

#include <immintrin.h>

#include <memory>
#include <random>

namespace gematria {

std::unique_ptr<Node, NodeDeleter> CreateRandomLinkedList(
    const std::size_t size) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, kMaxRandomListValue);
  Node *head = nullptr;
  for (int i = 0; i < size; i++) {
    head = new Node{.next = head, .value = distribution(generator)};
  }
  return std::unique_ptr<Node, NodeDeleter>(head);
}

void FlushLinkedListFromCache(const Node *ptr) {
  const Node *current = ptr;

  _mm_mfence();
  while (current) {
    const Node *temp = current->next;
    _mm_clflushopt(current);
    current = temp;
  }
  _mm_mfence();
}

}  // namespace gematria
