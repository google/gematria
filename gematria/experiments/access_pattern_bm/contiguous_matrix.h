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

#ifndef GEMATRIA_EXPERIMENTS_ACCESS_PATTERN_BM_CONTIGUOUS_MATRIX_H_
#define GEMATRIA_EXPERIMENTS_ACCESS_PATTERN_BM_CONTIGUOUS_MATRIX_H_

#include <memory>

namespace gematria {

std::unique_ptr<int[]>CreateRandomContiguousMatrix(std::size_t size);
void FlushContiguousMatrixFromCache(std::unique_ptr<int[]> &matrix, std::size_t size);

}  // namespace gematria

#endif
