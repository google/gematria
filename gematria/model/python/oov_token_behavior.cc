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

#include "gematria/model/oov_token_behavior.h"

#include "pybind11/pybind11.h"

namespace gematria {
namespace {

namespace py = ::pybind11;

PYBIND11_MODULE(oov_token_behavior, m) {
  m.doc() = "Specification of model behavior when it encounters an OOV token.";

  py::class_<OutOfVocabularyTokenBehavior> oov_token_behavior(
      m, "OutOfVocabularyTokenBehavior");
  oov_token_behavior
      .def_static("return_error", &OutOfVocabularyTokenBehavior::ReturnError)
      .def_static("replace_with_token",
                  &OutOfVocabularyTokenBehavior::ReplaceWithToken,
                  py::arg("replacement_token"))
      .def_property_readonly("behavior_type",
                             &OutOfVocabularyTokenBehavior::behavior_type)
      .def_property_readonly("replacement_token",
                             &OutOfVocabularyTokenBehavior::replacement_token);

  py::enum_<OutOfVocabularyTokenBehavior::BehaviorType>(oov_token_behavior,
                                                        "BehaviorType")
      .value("RETURN_ERROR",
             OutOfVocabularyTokenBehavior::BehaviorType::kReturnError)
      .value("REPLACE_TOKEN",
             OutOfVocabularyTokenBehavior::BehaviorType::kReplaceToken)
      .export_values();
}

}  // namespace
}  // namespace gematria
