#pragma once

#include <concepts>
#include <cstdint>

namespace filter {

namespace concepts {

template <typename T>
concept sampler = requires(T s, const std::uint32_t seed) {
  typename T::random_number_generator_type;

  { T() } -> std::same_as<T>;
  { s.seed(seed) } -> std::same_as<void>;
  { s.random_number_generator() } -> std::same_as<typename T::random_number_generator_type&>;
};

}  // namespace concepts

}  // namespace filter
