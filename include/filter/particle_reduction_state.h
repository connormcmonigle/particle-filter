#pragma once
#include <cstddef>
#include <cstdint>

namespace filter {

template <typename T>
struct particle_reduction_state {
  T most_likely_particle_;
  std::uint32_t count_;

  __host__ __device__ [[nodiscard]] constexpr const T& most_likely_particle() const noexcept { return most_likely_particle_; }
  __host__ __device__ [[nodiscard]] constexpr const std::uint32_t& count() const noexcept { return count_; }

  __host__ __device__ [[nodiscard]] static constexpr particle_reduction_state<T> zero() noexcept {
    return particle_reduction_state<T>{T{}, std::uint32_t{}};
  }

  __host__ __device__ [[nodiscard]] static constexpr particle_reduction_state<T> from_one_particle(const T& particle) noexcept {
    return particle_reduction_state<T>{particle, std::uint32_t{1}};
  }
};

template <typename T>
struct particle_reduction_state_transform {
  __host__ __device__ [[nodiscard]] inline particle_reduction_state<T> operator()(const T& particle) noexcept {
    return particle_reduction_state<T>::from_one_particle(particle);
  }
};

}  // namespace filter
