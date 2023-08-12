#pragma once
#include <cstddef>

namespace filter {

template <typename T>
struct particle_reduction_state {
  T most_likely_particle_;
  std::size_t count_;

  __host__ __device__ [[nodiscard]] constexpr const T& most_likely_particle() const noexcept { return most_likely_particle_; }
  __host__ __device__ [[nodiscard]] constexpr const std::size_t& count() const noexcept { return count_; }

  __host__ __device__ [[nodiscard]] static constexpr particle_reduction_state<T> zero() noexcept {
    return particle_reduction_state<T>{T{}, std::size_t{}};
  }

  __host__ __device__ [[nodiscard]] static constexpr particle_reduction_state<T> from_one_particle(const T& particle) noexcept {
    return particle_reduction_state<T>{particle, std::size_t{1}};
  }
};

template <typename T>
struct particle_reduction_state_transform {
  __host__ __device__ [[nodiscard]] inline particle_reduction_state<T> operator()(const T& particle) noexcept {
    return particle_reduction_state<T>::from_one_particle(particle);
  }
};

}  // namespace filter
