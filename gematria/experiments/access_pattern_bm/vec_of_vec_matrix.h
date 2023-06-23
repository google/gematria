#ifndef ACCESS_PATTERN_BM_VEC_OF_VEC_MATRIX_H_
#define ACCESS_PATTERN_BM_VEC_OF_VEC_MATRIX_H_

#include <vector>

namespace gematria {

std::vector<std::vector<int>> *CreateRandomVecOfVecMatrix(
    const std::size_t size);
void FlushVecOfVecMatrixFromCache(std::vector<std::vector<int>> *matrix);
void DeleteVecOfVecMatrix(std::vector<std::vector<int>> *matrix);

}  // namespace gematria

#endif