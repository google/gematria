#Experiment - access pattern benchmarks

This directory contains a set of microbenchmarks used to prototype data sets of 
basic blocks and functions augmented with cache hit and miss ratios.

### Build

The benchmarks can be built using
```bash
bazel build -c opt //gematria/experiments/access_pattern_bm/...
```
or, if timing measurements between flushing and non-flushing benchmarks are to 
be compared, using
```bash
bazel build -c opt --define balance_flushing_time=true //gematria/experiments/access_pattern_bm/...
```
and built binaries for each benchmark can be found down the corresponding 
path in `bazel-bin`.

### Benchmarks

Currently, there are microbenchmarks for the following access patterns:

 * Pointer chasing (linked lists, graphs): `linked_list_bm`,
 * Contiguous chunks of memory: `contiguous_matrix_bm`,
 * Vector-of-vector type accesses: `vec_of_vec_matrix_bm`,
 * Iterating over any STL-like container (multiset, list, deque, etc): `stl_container_bm`,
 * Iterating over any STL-like associative container (map, unordered_map, etc): `stl_container_bm`.
