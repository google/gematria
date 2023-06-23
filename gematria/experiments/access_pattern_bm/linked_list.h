#ifndef ACCESS_PATTERN_BM_LINKED_LIST_H_
#define ACCESS_PATTERN_BM_LINKED_LIST_H_

#include <iostream>

namespace gematria {

// Represents a linked list node
struct Node {
  Node *next = nullptr;
  int value;
};

Node *CreateRandomLinkedList(const std::size_t size);
void DeleteLinkedList(Node *ptr);
void FlushNodeFromCache(Node *ptr);
void FlushLinkedListFromCache(Node *ptr);

#endif

}  // namespace gematria
