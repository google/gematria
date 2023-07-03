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

#ifndef GEMATRIA_EXPERIMENTS_ACCESS_PATTERN_BM_LINKED_LIST_H_
#define GEMATRIA_EXPERIMENTS_ACCESS_PATTERN_BM_LINKED_LIST_H_

#include <memory>

namespace gematria {

// Represents a linked list node
struct Node {
  Node *next = nullptr;
  int value;
};

std::unique_ptr<Node, void (*)(Node *)> CreateRandomLinkedList(std::size_t size);
void FlushLinkedListFromCache(std::unique_ptr<Node, void (*)(Node *)> &ptr);

#endif

}  // namespace gematria
