#pragma once

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform_scan.h>
#include <thrust/tuple.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace filter {

template <typename T, typename std::enable_if<std::is_integral<T>::value, T>::type* = nullptr>
struct truncated_representation {
  T integral_component_;
  float partial_component_;

  __host__ __device__ [[nodiscard]] constexpr operator T() const noexcept { return integral_component_; }
  __host__ __device__ [[nodiscard]] constexpr const T& integral_component() const noexcept { return integral_component_; }
  __host__ __device__ [[nodiscard]] constexpr const float& partial_component() const noexcept { return partial_component_; }

  __host__ __device__ [[nodiscard]] static inline truncated_representation<T> from_value(const float& value) noexcept {
    const float truncated_value = floor(value);
    const T integral_component = static_cast<T>(truncated_value);
    const float partial_component = value - truncated_value;
    return truncated_representation<T>{integral_component, partial_component};
  }

  __host__ __device__ [[nodiscard]] static inline truncated_representation<T>
  bounded_plus(const T& bound, const truncated_representation<T>& a, const truncated_representation<T>& b) noexcept {
    const auto c = truncated_representation<T>::from_value(a.partial_component() + b.partial_component());
    const T integral_component = a.integral_component() + b.integral_component() + c.integral_component();
    const float partial_component = c.partial_component();
    return truncated_representation<T>{thrust::min(bound, integral_component), partial_component};
  }

  __host__ __device__ [[nodiscard]] static constexpr truncated_representation<T> from_integral(const T& value) noexcept {
    return truncated_representation<T>{value, float{}};
  }

  __host__ __device__ [[nodiscard]] static constexpr truncated_representation<T> zero() noexcept {
    return truncated_representation<T>{T{}, float{}};
  }
};

template <typename T>
struct scaled_truncated_representation_transform {
  float scale_;

  __host__ __device__ [[nodiscard]] inline truncated_representation<T> operator()(const float& value) noexcept {
    return truncated_representation<T>::from_value(scale_ * value);
  }

  constexpr scaled_truncated_representation_transform(const float& scale) noexcept : scale_{scale} {}
};

template <typename T>
struct truncated_representation_bounded_plus {
  T bound_;

  __host__ __device__ [[nodiscard]] inline truncated_representation<T> operator()(
      const truncated_representation<T>& a,
      const truncated_representation<T>& b) const noexcept {
    return truncated_representation<T>::bounded_plus(bound_, a, b);
  }

  constexpr truncated_representation_bounded_plus(const T& bound) noexcept : bound_{bound} {}
};

template <typename T, typename I>
class systematic_resampler {
  using weight_type = float;
  using particle_type = T;

  using index_type = I;
  using truncated_representation_type = truncated_representation<I>;

 private:
  std::size_t number_of_particles_;
  thrust::device_vector<particle_type> temp_particles_;
  thrust::device_vector<index_type> temp_particle_indices_;

  thrust::device_vector<weight_type> particle_weights_;
  thrust::device_vector<truncated_representation_type> particle_scatter_indices_;

 public:
  void resample(const thrust::device_vector<weight_type>& log_weights, thrust::device_vector<T>& particles) noexcept {
    const weight_type maximum_log_weight = thrust::reduce(
        log_weights.cbegin(), log_weights.cend(), -std::numeric_limits<weight_type>::infinity(), thrust::maximum<weight_type>());

    thrust::transform(
        thrust::device,
        log_weights.cbegin(),
        log_weights.cend(),
        particle_weights_.begin(),
        [maximum_log_weight] __device__(const weight_type& log_weight) { return __expf(log_weight - maximum_log_weight); });

    const weight_type cumulative_particle_weight =
        thrust::reduce(thrust::device, particle_weights_.cbegin(), particle_weights_.cend());

    const weight_type index_scale = static_cast<weight_type>(number_of_particles_) / cumulative_particle_weight;

    thrust::transform_exclusive_scan(
        thrust::device,
        particle_weights_.cbegin(),
        particle_weights_.cend(),
        particle_scatter_indices_.begin(),
        scaled_truncated_representation_transform<index_type>(index_scale),
        truncated_representation_type::zero(),
        truncated_representation_bounded_plus<index_type>(number_of_particles_));

    thrust::fill(thrust::device, temp_particle_indices_.begin(), temp_particle_indices_.end(), index_type{});

    const auto one = static_cast<index_type>(1);
    const auto input_index_iterator = thrust::make_counting_iterator<index_type>(index_type{});

    thrust::scatter_if(
        thrust::device,
        thrust::make_zip_iterator(input_index_iterator + one, particles.cbegin()),
        thrust::make_zip_iterator(input_index_iterator + number_of_particles_ + one, particles.cend()),
        thrust::make_transform_iterator(
            particle_scatter_indices_.cbegin(),
            [] __host__ __device__(const truncated_representation_type& index) { return index.integral_component(); }),
        thrust::make_zip_iterator(particle_scatter_indices_.cbegin(), std::next(particle_scatter_indices_.cbegin())),
        thrust::make_zip_iterator(temp_particle_indices_.begin(), temp_particles_.begin()),
        [] __host__ __device__(const thrust::tuple<truncated_representation_type, truncated_representation_type>& tuple) {
          return tuple.template get<1>().integral_component() > tuple.template get<0>().integral_component();
        });

    thrust::inclusive_scan(
        thrust::device,
        thrust::make_zip_iterator(temp_particle_indices_.begin(), temp_particles_.begin()),
        thrust::make_zip_iterator(temp_particle_indices_.end(), temp_particles_.end()),
        thrust::make_zip_iterator(temp_particle_indices_.begin(), temp_particles_.begin()),
        [] __host__ __device__(
            const thrust::tuple<index_type, particle_type>& a, const thrust::tuple<index_type, particle_type>& b) {
          return (a.template get<0>() > b.template get<0>()) ? a : b;
        });

    temp_particles_.swap(particles);
  }

  systematic_resampler(const std::size_t& number_of_particles) noexcept
      : number_of_particles_{number_of_particles},
        temp_particles_(number_of_particles),
        temp_particle_indices_(number_of_particles),
        particle_weights_(number_of_particles),
        particle_scatter_indices_(number_of_particles) {
    const auto last_scatter_index = truncated_representation_type::from_integral(number_of_particles);
    particle_scatter_indices_.push_back(last_scatter_index);
  }
};

}  // namespace filter
