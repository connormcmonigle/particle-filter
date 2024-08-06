#pragma once

#include <pf/config/target_config.h>

#include <cstddef>
#include <cstdint>

namespace pf::filter {

template <typename T>
struct particle_reduction_state {
  T most_likely_particle_;
  std::uint32_t count_;

  PF_TARGET_ATTRS [[nodiscard]] constexpr const T& most_likely_particle() const noexcept { return most_likely_particle_; }
  PF_TARGET_ATTRS [[nodiscard]] constexpr const std::uint32_t& count() const noexcept { return count_; }

  PF_TARGET_ATTRS [[nodiscard]] static constexpr particle_reduction_state<T> zero() noexcept {
    return particle_reduction_state<T>{T{}, std::uint32_t{}};
  }

  PF_TARGET_ATTRS [[nodiscard]] static constexpr particle_reduction_state<T> from_one_particle(const T& particle) noexcept {
    return particle_reduction_state<T>{particle, std::uint32_t{1}};
  }
};

template <typename T>
struct particle_reduction_state_transform {
  PF_TARGET_ATTRS [[nodiscard]] inline particle_reduction_state<T> operator()(const T& particle) noexcept {
    return particle_reduction_state<T>::from_one_particle(particle);
  }
};

}  // namespace pf::filter
