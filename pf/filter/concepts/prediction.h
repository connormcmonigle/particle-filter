#pragma once

#include <concepts>

namespace pf::filter::concepts {

template <typename T>
concept prediction = requires(const T p, const float t) {
  { T() } -> std::same_as<T>;
  { p.extrapolate_state(t) } -> std::same_as<T>;
};

}  // namespace pf::filter::concepts
