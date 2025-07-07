#pragma once

#include <pf/config/target_config.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform_scan.h>
#include <thrust/tuple.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>

namespace pf::filter {

template <typename T>
  requires std::unsigned_integral<T>
struct truncated_representation {
  T integral_component_;
  float partial_component_;

  PF_TARGET_ATTRS [[nodiscard]] constexpr operator T() const noexcept { return integral_component_; }
  PF_TARGET_ATTRS [[nodiscard]] constexpr const T& integral_component() const noexcept { return integral_component_; }
  PF_TARGET_ATTRS [[nodiscard]] constexpr const float& partial_component() const noexcept { return partial_component_; }

  PF_TARGET_ATTRS [[nodiscard]] static inline truncated_representation<T> from_value(const float& value) noexcept {
    const float truncated_value = floor(value);
    const T integral_component = static_cast<T>(truncated_value);
    const float partial_component = value - truncated_value;
    return truncated_representation<T>{integral_component, partial_component};
  }

  PF_TARGET_ATTRS [[nodiscard]] static inline truncated_representation<T>
  bounded_plus(const T& bound, const truncated_representation<T>& a, const truncated_representation<T>& b) noexcept {
    const auto c = truncated_representation<T>::from_value(a.partial_component() + b.partial_component());
    const T integral_component = a.integral_component() + b.integral_component() + c.integral_component();
    const float partial_component = c.partial_component();
    return truncated_representation<T>{thrust::min(bound, integral_component), partial_component};
  }

  PF_TARGET_ATTRS [[nodiscard]] static constexpr truncated_representation<T> from_integral(const T& value) noexcept {
    return truncated_representation<T>{value, float{}};
  }

  PF_TARGET_ATTRS [[nodiscard]] static constexpr truncated_representation<T> zero() noexcept {
    return truncated_representation<T>{T{}, float{}};
  }
};

template <typename T, typename Iterator>
  requires std::unsigned_integral<T>
struct scaled_truncated_representation_transform {
  float number_of_particles_;
  Iterator total_particle_weight_iterator_;

  PF_TARGET_ATTRS [[nodiscard]] inline truncated_representation<T> operator()(const float& value) noexcept {
    return truncated_representation<T>::from_value((number_of_particles_ / *total_particle_weight_iterator_) * value);
  }

  constexpr scaled_truncated_representation_transform(
      const float& number_of_particles,
      const Iterator& total_particle_weight_iterator) noexcept
      : number_of_particles_{number_of_particles}, total_particle_weight_iterator_{total_particle_weight_iterator} {}
};

template <typename T, typename Iterator>
  requires std::unsigned_integral<T>
inline constexpr scaled_truncated_representation_transform<T, Iterator> make_scaled_truncated_representation_transform(
    const std::size_t& number_of_particles,
    const Iterator& total_particle_weight_iterator) {
  return scaled_truncated_representation_transform<T, Iterator>(
      static_cast<float>(number_of_particles), total_particle_weight_iterator);
}

template <typename T>
  requires std::unsigned_integral<T>
struct truncated_representation_bounded_plus {
  T bound_;

  PF_TARGET_ATTRS [[nodiscard]] inline truncated_representation<T> operator()(
      const truncated_representation<T>& a,
      const truncated_representation<T>& b) const noexcept {
    return truncated_representation<T>::bounded_plus(bound_, a, b);
  }

  constexpr truncated_representation_bounded_plus(const T& bound) noexcept : bound_{bound} {}
};

template <typename T, typename I>
  requires std::unsigned_integral<I>
class systematic_resampler {
  using weight_type = float;
  using particle_type = T;

  using index_type = I;
  using truncated_representation_type = truncated_representation<I>;

 private:
  std::size_t number_of_particles_;
  target_config::vector<particle_type> temp_particles_;
  target_config::vector<index_type> temp_particle_indices_;

  target_config::vector<weight_type> particle_weights_;
  target_config::vector<truncated_representation_type> particle_scatter_indices_;

  target_config::vector<weight_type> maximum_log_weight_single_;
  target_config::vector<weight_type> total_particle_weight_single_;

 public:
  template <typename ExecutionPolicy>
  void resample(
      const ExecutionPolicy& execution_policy,
      const target_config::vector<weight_type>& log_weights,
      target_config::vector<T>& particles) noexcept {
    const auto reduce_index_key_iterator = thrust::make_constant_iterator<index_type>(0);

    const auto maximum_log_weight_iterator = maximum_log_weight_single_.cbegin();
    const auto total_particle_weight_iterator = total_particle_weight_single_.cbegin();

    thrust::reduce_by_key(
        execution_policy,
        reduce_index_key_iterator,
        reduce_index_key_iterator + number_of_particles_,
        log_weights.cbegin(),
        thrust::make_discard_iterator(),
        maximum_log_weight_single_.begin(),
        thrust::equal_to<index_type>(),
        thrust::maximum<weight_type>());

    thrust::transform(
        execution_policy,
        log_weights.cbegin(),
        log_weights.cend(),
        particle_weights_.begin(),
        [maximum_log_weight_iterator] PF_TARGET_ATTRS(const weight_type& log_weight) {
          return expf(log_weight - *maximum_log_weight_iterator);
        });

    thrust::reduce_by_key(
        execution_policy,
        reduce_index_key_iterator,
        reduce_index_key_iterator + number_of_particles_,
        particle_weights_.cbegin(),
        thrust::make_discard_iterator(),
        total_particle_weight_single_.begin(),
        thrust::equal_to<index_type>());

    thrust::transform_exclusive_scan(
        execution_policy,
        particle_weights_.cbegin(),
        particle_weights_.cend(),
        particle_scatter_indices_.begin(),
        make_scaled_truncated_representation_transform<index_type>(number_of_particles_, total_particle_weight_iterator),
        truncated_representation_type::zero(),
        truncated_representation_bounded_plus<index_type>(number_of_particles_));

    thrust::fill(execution_policy, temp_particle_indices_.begin(), temp_particle_indices_.end(), index_type{});

    const auto one = static_cast<index_type>(1);
    const auto input_index_iterator = thrust::make_counting_iterator<index_type>(index_type{});

    thrust::scatter_if(
        execution_policy,
        thrust::make_zip_iterator(input_index_iterator + one, particles.cbegin()),
        thrust::make_zip_iterator(input_index_iterator + number_of_particles_ + one, particles.cend()),
        thrust::make_transform_iterator(
            particle_scatter_indices_.cbegin(),
            [] PF_TARGET_ATTRS(const truncated_representation_type& index) { return index.integral_component(); }),
        thrust::make_zip_iterator(particle_scatter_indices_.cbegin(), std::next(particle_scatter_indices_.cbegin())),
        thrust::make_zip_iterator(temp_particle_indices_.begin(), temp_particles_.begin()),
        [] PF_TARGET_ATTRS(const thrust::tuple<truncated_representation_type, truncated_representation_type>& tuple) {
          return thrust::get<1>(tuple).integral_component() > thrust::get<0>(tuple).integral_component();
        });

    thrust::inclusive_scan(
        execution_policy,
        thrust::make_zip_iterator(temp_particle_indices_.begin(), temp_particles_.begin()),
        thrust::make_zip_iterator(temp_particle_indices_.end(), temp_particles_.end()),
        thrust::make_zip_iterator(temp_particle_indices_.begin(), temp_particles_.begin()),
        [] PF_TARGET_ATTRS(const thrust::tuple<index_type, particle_type>& a, const thrust::tuple<index_type, particle_type>& b) {
          return (thrust::get<0>(a) > thrust::get<0>(b)) ? a : b;
        });

    temp_particles_.swap(particles);
  }

  systematic_resampler(const std::size_t& number_of_particles) noexcept
      : number_of_particles_{number_of_particles},
        temp_particles_(number_of_particles),
        temp_particle_indices_(number_of_particles),
        particle_weights_(number_of_particles),
        particle_scatter_indices_(number_of_particles),
        maximum_log_weight_single_(1),
        total_particle_weight_single_(1) {
    const auto last_scatter_index = truncated_representation_type::from_integral(number_of_particles);
    particle_scatter_indices_.push_back(last_scatter_index);
  }
};

}  // namespace pf::filter
