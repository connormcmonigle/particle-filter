#pragma once
#include <cstddef>

namespace filter {

template <typename T>
struct particle_reduction_state {
  T most_likely_particle_;
  std::size_t count_;

   [[nodiscard]] constexpr const T& most_likely_particle() const noexcept { return most_likely_particle_; }
   [[nodiscard]] constexpr const std::size_t& count() const noexcept { return count_; }

   [[nodiscard]] static constexpr particle_reduction_state<T> zero() noexcept {
    return particle_reduction_state<T>{T{}, std::size_t{}};
  }

   [[nodiscard]] static constexpr particle_reduction_state<T> from_one_particle(const T& particle) noexcept {
    return particle_reduction_state<T>{particle, std::size_t{1}};
  }
};

template <typename T>
struct particle_reduction_state_transform {
   [[nodiscard]] inline particle_reduction_state<T> operator()(const T& particle) noexcept {
    return particle_reduction_state<T>::from_one_particle(particle);
  }
};

}  // namespace filter
