#include "linked_list.h"

#include <immintrin.h>

#include <random>

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