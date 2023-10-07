// Copyright 2022 Google Inc.
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

// Defines a class that describes the behavior of token-based models in
// situations when they encounter tokens not known during development time.

#ifndef GEMATRIA_MODEL_OOV_TOKEN_BEHAVIOR_H_
#define GEMATRIA_MODEL_OOV_TOKEN_BEHAVIOR_H_

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

namespace gematria {

// Specification of the behavior of the token builder class when it encounters
// an out of vocabulary token.
class OutOfVocabularyTokenBehavior {
 public:
  // The types of behavior when the graph builder encounters a token that is not
  // in its vocabulary.
  enum class BehaviorType {
    // The basic block that contains the token is not added to the graph;
    // GraphBuilder::AddBasicBlock returns an error.
    kReturnError = 0,
    // The unknown token is replaced by a specific (known) token specified
    // during the construction of the graph builder.
    kReplaceToken = 1,
  };

  // A behavior that makes the graph builder return an error status when it
  // encounters an out of vocabulary token.
  static OutOfVocabularyTokenBehavior ReturnError() {
    return OutOfVocabularyTokenBehavior(BehaviorType::kReturnError,
                                        std::string());
  }

  // A behavior that makes the graph builder replace out of vocabulary tokens
  // with `replacement_token`. The replacement token must be part of the
  // vocabulary.
  static OutOfVocabularyTokenBehavior ReplaceWithToken(
      std::string replacement_token) {
    if (replacement_token.empty()) {
      // TODO(ondrasej): Make the method retrun a status.
      std::cerr << "The replacement token must not be empty.";
      std::abort();
    }
    return OutOfVocabularyTokenBehavior(BehaviorType::kReplaceToken,
                                        std::move(replacement_token));
  }

  BehaviorType behavior_type() const { return behavior_type_; }
  const std::string& replacement_token() const { return replacement_token_; }

 private:
  // Instances of this class should not be created directly. Instead, use one of
  // the factory methods above.
  OutOfVocabularyTokenBehavior(BehaviorType behavior_type,
                               std::string replacement_token)
      : behavior_type_(behavior_type),
        replacement_token_(std::move(replacement_token)) {}

  BehaviorType behavior_type_;
  std::string replacement_token_;
};

}  // namespace gematria

#endif  // GEMATRIA_MODEL_OOV_TOKEN_BEHAVIOR_H_
