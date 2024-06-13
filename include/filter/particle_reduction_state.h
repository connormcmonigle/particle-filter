#pragma once

#include <config/target_config.h>

#include <cstddef>
#include <cstdint>

namespace filter {

template <typename T>
struct particle_reduction_state {
  T most_likely_particle_;
  std::uint32_t count_;

  PARTICLE_FILTER_TARGET_ATTRS [[nodiscard]] constexpr const T& most_likely_particle() const noexcept {
    return most_likely_particle_;
  }
  PARTICLE_FILTER_TARGET_ATTRS [[nodiscard]] constexpr const std::uint32_t& count() const noexcept { return count_; }

  PARTICLE_FILTER_TARGET_ATTRS [[nodiscard]] static constexpr particle_reduction_state<T> zero() noexcept {
    return particle_reduction_state<T>{T{}, std::uint32_t{}};
  }

  PARTICLE_FILTER_TARGET_ATTRS [[nodiscard]] static constexpr particle_reduction_state<T> from_one_particle(
      const T& particle) noexcept {
    return particle_reduction_state<T>{particle, std::uint32_t{1}};
  }
};

template <typename T>
struct particle_reduction_state_transform {
  PARTICLE_FILTER_TARGET_ATTRS [[nodiscard]] inline particle_reduction_state<T> operator()(const T& particle) noexcept {
    return particle_reduction_state<T>::from_one_particle(particle);
  }
};

}  // namespace filter
