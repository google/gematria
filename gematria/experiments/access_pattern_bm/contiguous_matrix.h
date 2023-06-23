#ifndef ACCESS_PATTERN_BM_CONTIGUOUS_MATRIX_H_
#define ACCESS_PATTERN_BM_CONTIGUOUS_MATRIX_H_

#include <iostream>

namespace gematria {

int *CreateRandomContiguousMatrix(const std::size_t size);
void FlushContiguousMatrixFromCache(int *matrix, std::size_t size);
void DeleteContiguousMatrix(int *matrix);

}  // namespace gematria

#endif