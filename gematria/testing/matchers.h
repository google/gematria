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

#ifndef THIRD_PARTY_GEMATRIA_GEMATRIA_TESTING_MATCHERS_H_
#define THIRD_PARTY_GEMATRIA_GEMATRIA_TESTING_MATCHERS_H_

#include <initializer_list>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"

namespace gematria {

// Returns true when left and right contain the same text, save for whitespace.
// 1. ignores leading and trailing whitespace.
// 2. non-whitespace characters are compared for equality.
// 3. all whitespace inside the strings is considered to be equal, regardless of
//    the number and type of whitespace characters.
// 4. `left` and `right` must contain whitespace blocks in the same positions.
//
// Examples:
//   "foo" is equal to "foo",
//   "foo" is NOT equal to "bar",
//   "foo" is equal to "   foo   ",
//   "foo bar" is equal to "foo  \t\t\n   bar",
//   "foo bar" is NOT equal to "foobar".
bool EqualsNormalizingWhitespace(std::string_view left, std::string_view right);

testing::Matcher<std::string> EqualsNormalizingWhitespace(std::string other);

namespace internal {

void AddIgnoredFieldsToDifferencer(
    const ::google::protobuf::Descriptor* descriptor,
    const std::vector<std::string>& ignored_field_names,
    ::google::protobuf::util::MessageDifferencer* differencer);

template <typename ProtoType>
bool MatchProto(const ProtoType& actual_proto,
                const std::string& expected_proto_str,
                const std::vector<std::string>& ignored_fields,
                ::google::protobuf::util::MessageDifferencer::Scope scope,
                ::testing::MatchResultListener* listener) {
  using ::google::protobuf::TextFormat;
  using ::google::protobuf::util::MessageDifferencer;
  ProtoType expected_proto;
  if (!TextFormat::ParseFromString(expected_proto_str, &expected_proto)) {
    *listener << "could not parse proto: <" << expected_proto_str << ">";
    return false;
  }

  MessageDifferencer differencer;
  std::string differences;
  differencer.ReportDifferencesToString(&differences);
  differencer.set_scope(scope);
  AddIgnoredFieldsToDifferencer(expected_proto.descriptor(), ignored_fields,
                                &differencer);
  if (!differencer.Compare(expected_proto, actual_proto)) {
    *listener << "the protos are different:\n" << differences;
    return false;
  }

  return true;
}

}  // namespace internal

// A gMock matcher that takes a proto in the text format and compares input
// protos against this text representation. Used to implement EqualsProto(str).
class EqualsProtoMatcher {
 public:
  explicit EqualsProtoMatcher(std::string expected_proto_str)
      : expected_proto_str_(std::move(expected_proto_str)) {}

  template <typename ProtoType>
  bool MatchAndExplain(const ProtoType& actual_proto,
                       ::testing::MatchResultListener* listener) const {
    return internal::MatchProto(actual_proto, expected_proto_str_,
                                ignored_fields_, scope_, listener);
  }

  void DescribeTo(std::ostream* os) const {
    *os << "equals to proto:\n" << expected_proto_str_;
  }

  void DescribeNegationTo(std::ostream* os) const {
    *os << "is not equal to proto:\n" << expected_proto_str_;
  }

  template <typename Iterator>
  void AddIgnoredFields(Iterator begin, Iterator end) {
    ignored_fields_.insert(ignored_fields_.end(), begin, end);
  }

  void SetComparePartially() {
    scope_ = ::google::protobuf::util::MessageDifferencer::PARTIAL;
  }

 private:
  const std::string expected_proto_str_;
  ::google::protobuf::util::MessageDifferencer::Scope scope_ =
      ::google::protobuf::util::MessageDifferencer::FULL;
  std::vector<std::string> ignored_fields_;
};

// Creates a polymorphic proto matcher based on the given proto in text format.
inline ::testing::PolymorphicMatcher<EqualsProtoMatcher> EqualsProto(
    std::string_view expected_proto_str) {
  return ::testing::MakePolymorphicMatcher(
      EqualsProtoMatcher(std::string(expected_proto_str)));
}

// Creates a polymorphic proto matcher based on the given proto.
inline ::testing::PolymorphicMatcher<EqualsProtoMatcher> EqualsProto(
    const google::protobuf::Message& expected_proto) {
  std::string expected_proto_str;
  using ::google::protobuf::TextFormat;
  CHECK(TextFormat::PrintToString(expected_proto, &expected_proto_str));
  return ::testing::MakePolymorphicMatcher(
      EqualsProtoMatcher(std::move(expected_proto_str)));
}

template <typename InnerProtoMatcher, typename StringType>
InnerProtoMatcher IgnoringFields(
    const std::initializer_list<StringType>& fields,
    InnerProtoMatcher matcher) {
  matcher.mutable_impl().AddIgnoredFields(fields.begin(), fields.end());
  return matcher;
}

template <typename InnerProtoMatcher>
InnerProtoMatcher Partially(InnerProtoMatcher matcher) {
  matcher.mutable_impl().SetComparePartially();
  return matcher;
}

// A gMock matcher that takes a tuple containing a proto and a string containing
// a proto in text format, and compares the proto against the text
// representation. Used to implement EqualsProto().
class EqualsProtoTupleMatcher {
 public:
  EqualsProtoTupleMatcher() {}

  template <typename TupleType>
  bool MatchAndExplain(TupleType args,
                       ::testing::MatchResultListener* listener) const {
    using ::testing::get;
    // TODO(ondrasej): Add support for ignored fields when needed.
    return internal::MatchProto(
        get<0>(args), get<1>(args), {},
        ::google::protobuf::util::MessageDifferencer::FULL, listener);
  }

  void DescribeTo(std::ostream* os) const { *os << "are equal"; }

  void DescribeNegationTo(std::ostream* os) const { *os << "is not equal"; }
};

// Creates a tuple-based proto matcher that can be used e.g. with
// ::testing::Pointwise.
inline ::testing::PolymorphicMatcher<EqualsProtoTupleMatcher> EqualsProto() {
  return ::testing::MakePolymorphicMatcher(EqualsProtoTupleMatcher());
}

// Implements IsOk() as a polymorphic matcher.
MATCHER(IsOk, "") { return arg.ok(); }

#ifndef ASSERT_OK
#define ASSERT_OK(status_expr) EXPECT_THAT(status_expr, ::gematria::IsOk())
#endif  // ASSERT_OK
#ifndef EXPECT_OK
#define EXPECT_OK(status_expr) EXPECT_THAT(status_expr, ::gematria::IsOk())
#endif  // EXPECT_OK

namespace internal {

// Monomorphic matcher for the error code of a absl::Status.
inline bool StatusIsMatcher(const absl::Status& actual_status,
                            const absl::StatusCode& expected_error_code) {
  return actual_status.code() == expected_error_code;
}

// Monomorphic matcher for the error code of a StatusOr.
template <typename T>
inline bool StatusIsMatcher(const absl::StatusOr<T>& actual_status_or,
                            const absl::StatusCode& expected_error_code) {
  return StatusIsMatcher(actual_status_or.status(), expected_error_code);
}

// Monomorphic matcher for the error code & message of a absl::Status.
inline bool StatusIsMatcher(
    const absl::Status& actual_status,
    const absl::StatusCode& expected_error_code,
    const ::testing::Matcher<std::string_view>& expected_message) {
  ::testing::StringMatchResultListener sink;
  return actual_status.code() == expected_error_code &&
         expected_message.MatchAndExplain(actual_status.message(), &sink);
}

// Monomorphic matcher for the error code & message of a StatusOr.
template <typename T>
inline bool StatusIsMatcher(
    const absl::StatusOr<T>& actual_status_or,
    const absl::StatusCode& expected_error_code,
    const ::testing::Matcher<std::string_view>& expected_message) {
  return StatusIsMatcher(actual_status_or.status(), expected_error_code,
                         expected_message);
}

}  // namespace internal

// Implements StatusIs() as a polymorphic matcher.
MATCHER_P(StatusIs, expected_error_code, "") {
  return internal::StatusIsMatcher(arg, expected_error_code);
}

// Implements StatusIs() as a polymorphic matcher.
MATCHER_P2(StatusIs, expected_error_code, expected_message, "") {
  return internal::StatusIsMatcher(arg, expected_error_code, expected_message);
}

namespace internal {

// Monomorphic implementation of a matcher for a StatusOr.
template <typename StatusOrType>
class IsOkAndHoldsMatcherImpl
    : public ::testing::MatcherInterface<StatusOrType> {
 public:
  using ValueType = typename std::remove_reference<
      decltype(std::declval<StatusOrType>().value())>::type;

  template <typename InnerMatcher>
  explicit IsOkAndHoldsMatcherImpl(InnerMatcher&& inner_matcher)
      : inner_matcher_(::testing::SafeMatcherCast<const ValueType&>(
            std::forward<InnerMatcher>(inner_matcher))) {}

  void DescribeTo(std::ostream* os) const {
    *os << "is OK and has a value that ";
    inner_matcher_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const {
    *os << "isn't OK or has a value that ";
    inner_matcher_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(StatusOrType actual_value,
                       ::testing::MatchResultListener* listener) const {
    if (!actual_value.ok()) {
      *listener << "which has status " << actual_value.status();
      return false;
    }

    ::testing::StringMatchResultListener inner_listener;
    const bool matches =
        inner_matcher_.MatchAndExplain(actual_value.value(), &inner_listener);
    const std::string inner_explanation = inner_listener.str();
    if (!inner_explanation.empty()) {
      *listener << "which contains value "
                << ::testing::PrintToString(actual_value.value()) << ", "
                << inner_explanation;
    }
    return matches;
  }

 private:
  const ::testing::Matcher<const ValueType&> inner_matcher_;
};

// Implements IsOkAndHolds() as a polymorphic matcher.
template <typename InnerMatcher>
class IsOkAndHoldsMatcher {
 public:
  explicit IsOkAndHoldsMatcher(InnerMatcher inner_matcher)
      : inner_matcher_(std::move(inner_matcher)) {}

  // Converts this polymorphic matcher to a monomorphic one of the given type.
  // StatusOrType can be either StatusOr<T> or a reference to StatusOr<T>.
  template <typename StatusOrType>
  operator ::testing::Matcher<StatusOrType>() const {
    return ::testing::MakeMatcher(
        new IsOkAndHoldsMatcherImpl<StatusOrType>(inner_matcher_));
  }

 private:
  const InnerMatcher inner_matcher_;
};

}  // namespace internal

// Returns a gMock matcher that matches a StatusOr<> whose status is
// OK and whose value matches the inner matcher.
template <typename InnerMatcher>
internal::IsOkAndHoldsMatcher<typename std::decay<InnerMatcher>::type>
IsOkAndHolds(InnerMatcher&& inner_matcher) {
  return internal::IsOkAndHoldsMatcher<typename std::decay<InnerMatcher>::type>(
      std::forward<InnerMatcher>(inner_matcher));
}

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_TESTING_MATCHERS_H_
