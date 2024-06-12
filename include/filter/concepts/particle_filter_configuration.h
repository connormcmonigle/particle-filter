#pragma once

#include <filter/concepts/observation.h>
#include <filter/concepts/particle_reduction_operation.h>
#include <filter/concepts/prediction.h>
#include <util/random_variable_sampler.h>

#include <concepts>

namespace filter {

namespace concepts {

template <typename T>
concept particle_filter_configuration = requires(
    const T c,
    const float t,
    util::default_rv_sampler s,
    const typename T::observation_type o,
    typename T::prediction_type p) {
  requires observation<typename T::observation_type>;
  requires prediction<typename T::prediction_type>;

  { c.apply_process(t, s, p) } -> std::same_as<void>;
  { c.sample_from(s, o) } -> std::same_as<typename T::prediction_type>;
  { c.conditional_log_likelihood(std::as_const(s), o, p) } -> std::same_as<float>;
  { c.most_likely_particle_reduction() } -> particle_reduction_operation<typename T::prediction_type>;
};

}  // namespace concepts

}  // namespace filter
