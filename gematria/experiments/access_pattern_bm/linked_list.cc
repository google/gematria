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

#include <random>

namespace gematria {

static std::default_random_engine generator;
static std::uniform_int_distribution<int> distribution(0, 1023);

Node *CreateRandomLinkedList(const std::size_t size) {
  Node *head = new Node;
  Node *current = head;

  int i = size;
  while (i--) {
    current->next = new Node;
    current->value = distribution(generator);
    current = current->next;
  }

  return head;
}

void FlushNodeFromCache(Node *ptr) { _mm_clflushopt(ptr); }

void FlushLinkedListFromCache(Node *ptr) {
  _mm_mfence();
  while (ptr) {
    Node *temp = ptr->next;
    FlushNodeFromCache(ptr);
    ptr = temp;
  }
  _mm_mfence();
}

void DeleteLinkedList(Node *ptr) {
  while (ptr) {
    Node *temp = ptr->next;
    delete ptr;
    ptr = temp;
  }
}

}  // namespace gematria
