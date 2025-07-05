#pragma once

#include <pf/config/target_config.h>
#include <pf/filter/particle_reduction_state.h>
#include <point/observation.h>
#include <point/particle_filter_configuration_parameters.h>
#include <point/prediction.h>
#include <point/sampler.h>
#include <thrust/random.h>

#include <Eigen/Dense>

namespace point {

struct most_likely_particle_reduction_impl {
  using state_type = pf::filter::particle_reduction_state<prediction>;

  PF_TARGET_ATTRS [[nodiscard]] inline state_type operator()(const state_type& a, const state_type& b) const noexcept {
    const prediction a_particle = a.most_likely_particle();
    const prediction b_particle = b.most_likely_particle();

    const float alpha = static_cast<float>(b.count()) / static_cast<float>(a.count() + b.count());
    const float c_alpha = 1.0f - alpha;

    const Eigen::Vector3f position = c_alpha * a_particle.position() + alpha * b_particle.position();
    const Eigen::Vector3f velocity = c_alpha * a_particle.velocity() + alpha * b_particle.velocity();

    const auto state = prediction(position, velocity);
    return state_type{state, a.count() + b.count()};
  }
};

class particle_filter_configuration {
 private:
  Eigen::Vector3f velocity_prior_diagonal_covariance_;
  Eigen::Vector3f velocity_process_diagonal_covariance_;

 public:
  using observation_type = observation;
  using prediction_type = prediction;

  using sampler_type = sampler;

  [[nodiscard]] most_likely_particle_reduction_impl most_likely_particle_reduction() const noexcept {
    return most_likely_particle_reduction_impl{};
  }

  PF_TARGET_ATTRS [[nodiscard]] float
  conditional_log_likelihood(const sampler& sampler, const observation& state, const prediction& given) const noexcept {
    const Eigen::Vector3f error = state.position() - given.position();
    return sampler.unnormalized_normal_log_density(state.position_diagonal_covariance(), error);
  }

  PF_TARGET_ATTRS prediction sample_from(sampler& sampler, const observation& state) const noexcept {
    const auto position_noise = sampler.normal_sample(state.position_diagonal_covariance());
    const auto velocity_noise = sampler.normal_sample(velocity_prior_diagonal_covariance_);
    return prediction(state.position() + position_noise, velocity_noise);
  }

  PF_TARGET_ATTRS void apply_process(const float& time_offset_seconds, sampler& sampler, prediction& state) const noexcept {
    const auto noise_0 = sampler.normal_sample(velocity_process_diagonal_covariance_);
    const auto noise_1 = sampler.normal_sample(velocity_process_diagonal_covariance_);
    state.update_state(time_offset_seconds, noise_0, noise_1);
  }

  particle_filter_configuration(const particle_filter_configuration_parameters& params) noexcept
      : velocity_prior_diagonal_covariance_{params.velocity_prior_diagonal_covariance},
        velocity_process_diagonal_covariance_{params.velocity_process_diagonal_covariance} {}
};

}  // namespace point
