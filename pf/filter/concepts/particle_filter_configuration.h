#pragma once

#include <pf/filter/concepts/observation.h>
#include <pf/filter/concepts/particle_reduction_operation.h>
#include <pf/filter/concepts/prediction.h>
#include <pf/filter/concepts/sampler.h>

#include <concepts>

namespace pf::filter::concepts {

template <typename T>
concept particle_filter_configuration = requires(
    const T c,
    const float t,
    typename T::sampler_type s,
    const typename T::observation_type o,
    typename T::prediction_type p) {
  requires sampler<typename T::sampler_type>;
  requires observation<typename T::observation_type>;
  requires prediction<typename T::prediction_type>;

  { c.apply_process(t, s, p) } -> std::same_as<void>;
  { c.sample_from(s, o) } -> std::same_as<typename T::prediction_type>;
  { c.conditional_log_likelihood(std::as_const(s), o, p) } -> std::same_as<float>;
  { c.most_likely_particle_reduction() } -> particle_reduction_operation<typename T::prediction_type>;
};

}  // namespace pf::filter::concepts
