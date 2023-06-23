# Experiment - access pattern benchmarks

This directory contains a set of microbenchmarks whose purpose 
is to serve as a sanity check of sorts for cache hit/miss performance 
related predictions.

### Build

The benchmarks can be built using
```bash
bazel build //gematria/experiments/access_pattern_bm:all
```
and built binaries for each benchmark can be found down the corresponding 
path in `bazel-bin`.

### Benchmarks

Currently, there are microbenchmarks for the following access patterns:

 * Pointer chasing (linked lists, graphs): `linked_list_bm`
 * Contiguous chunks of memory: `contiguous_matrix_bm`
 * Vector-of-vector type accesses: `vec_of_vec_matrix_bm`
 * Iterating over any STL-like container (multiset, list, deque, etc): `stl_container_bm`
 * Iterating over any STL-like associative container (map, unordered_map, etc): `stl_container_bm`