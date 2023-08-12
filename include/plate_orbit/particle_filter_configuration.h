#pragma once

#include <filter/particle_reduction_state.h>
#include <plate_orbit/observation.h>
#include <plate_orbit/observed_plate.h>
#include <plate_orbit/observed_plate_orbit.h>
#include <plate_orbit/observed_plate_orbit_builder.h>
#include <plate_orbit/particle_filter_configuration_parameters.h>
#include <plate_orbit/predicted_plate.h>
#include <plate_orbit/prediction.h>
#include <thrust/random.h>
#include <util/device_array.h>
#include <util/random_variable_sampler.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace plate_orbit {

namespace helper {

__device__ [[nodiscard]] inline float log_sigmoid(const float& x) noexcept { return -__logf(1.0f + __expf(-x)); }

}  // namespace helper

struct most_likely_particle_reduction_impl {
  using state_type = filter::particle_reduction_state<prediction>;
  static constexpr float half_pi = M_PI_2;

  struct orientation_and_radii {
    float orientation;
    float radius_0;
    float radius_1;
  };

  __device__ [[nodiscard]] inline state_type operator()(const state_type& a, const state_type& b) const noexcept {
    // this does not necessarily provide the correct mean in the MLE sense,
    // though it gives a reasonable approximation for most inputs.

    const prediction a_particle = a.most_likely_particle();
    const prediction b_particle = b.most_likely_particle();

    const util::device_array<orientation_and_radii, 5> candidates = {
        orientation_and_radii{a_particle.orientation() + 0.0f * half_pi, a_particle.radius_0(), a_particle.radius_1()},
        orientation_and_radii{a_particle.orientation() + 1.0f * half_pi, a_particle.radius_1(), a_particle.radius_0()},
        orientation_and_radii{a_particle.orientation() - 1.0f * half_pi, a_particle.radius_1(), a_particle.radius_0()},
        orientation_and_radii{a_particle.orientation() + 2.0f * half_pi, a_particle.radius_0(), a_particle.radius_1()},
        orientation_and_radii{a_particle.orientation() - 2.0f * half_pi, a_particle.radius_0(), a_particle.radius_1()},
    };

    const auto target = b_particle.orientation();
    const auto [a_orientation, a_radius_0, a_radius_1] =
        *candidates.minimum_by([target](const auto& value) { return abs(target - value.orientation); });

    const float alpha = static_cast<float>(b.count()) / static_cast<float>(a.count() + b.count());
    const float c_alpha = 1.0f - alpha;

    const float radius_0 = c_alpha * a_radius_0 + alpha * b_particle.radius_0();
    const float radius_1 = c_alpha * a_radius_1 + alpha * b_particle.radius_1();

    const float orientation = c_alpha * a_orientation + alpha * b_particle.orientation();
    const float orientation_velocity = c_alpha * a_particle.orientation_velocity() + alpha * b_particle.orientation_velocity();

    const Eigen::Vector3f center = c_alpha * a_particle.center() + alpha * b_particle.center();
    const Eigen::Vector3f center_velocity = c_alpha * a_particle.center_velocity() + alpha * b_particle.center_velocity();

    const auto state = prediction(radius_0, radius_1, orientation, orientation_velocity, center, center_velocity);
    return state_type{state, a.count() + b.count()};
  }
};

class particle_filter_configuration {
 private:
  particle_filter_configuration_parameters params_;

 public:
  using observation_type = observation;
  using prediction_type = prediction;

  [[nodiscard]] most_likely_particle_reduction_impl most_likely_particle_reduction() const noexcept {
    return most_likely_particle_reduction_impl{};
  }

  __device__ [[nodiscard]] float conditional_log_likelihood(
      const util::default_rv_sampler& sampler,
      const observation& state,
      const prediction& given) const noexcept {
    const auto predicted_plates = given.predicted_plates().selection_sort_by(
        [state](const predicted_plate& a) { return (state.observer_position() - a.position()).squaredNorm(); });

    const auto [pred_0, pred_1, pred_2, pred_3] = predicted_plates;
    const Eigen::Vector3f observer_delta = state.observer_position() - given.center();

    const auto [logit_visibility_0, logit_visibility_1, logit_visibility_2, logit_visibility_3] =
        predicted_plates.transformed([&, this](const predicted_plate& plate) {
          const Eigen::Vector3f plate_delta = plate.position() - given.center();
          const float similarity = observer_delta.dot(plate_delta) / (observer_delta.norm() * plate_delta.norm());
          return params_.visibility_logit_coefficient * similarity;
        });

    auto log_p_of = [&sampler](const observed_plate& x, const predicted_plate& y) {
      const auto error = (x.position() - y.position()).eval();
      return sampler.unnormalized_normal_log_density(x.position_diagonal_covariance(), error);
    };

    if (!state.plate_two().has_value()) {
      const float pr_visibility = helper::log_sigmoid(logit_visibility_0) + helper::log_sigmoid(-logit_visibility_1) +
                                  helper::log_sigmoid(-logit_visibility_2) + helper::log_sigmoid(-logit_visibility_3);

      return pr_visibility + log_p_of(state.plate_one(), pred_0);
    }

    const float log_pr_visibility = helper::log_sigmoid(logit_visibility_0) + helper::log_sigmoid(logit_visibility_1) +
                                    helper::log_sigmoid(-logit_visibility_2) + helper::log_sigmoid(-logit_visibility_3);

    const float assignment_one = log_p_of(state.plate_one(), pred_0) + log_p_of(*state.plate_two(), pred_1);
    const float assignment_two = log_p_of(state.plate_one(), pred_1) + log_p_of(*state.plate_two(), pred_0);
    return log_pr_visibility + std::max(assignment_one, assignment_two);
  }

  __device__ [[nodiscard]] prediction sample_from(
      util::default_rv_sampler& sampler,
      const observation& state) const noexcept {
    const observed_plate_orbit_builder builder(params_.radius_prior, state.observer_position());

    const observed_plate_orbit orbit = state.plate_two().has_value() ?
                                           builder.from_two_plates(state.plate_one(), *state.plate_two()) :
                                           builder.from_one_plate(state.plate_one());

    const float radius_variance =
        state.plate_two().has_value() ? params_.radius_prior_variance_two_plates : params_.radius_prior_variance_one_plate;

    const float radius_0 = orbit.radius + sampler.normal_sample(radius_variance);
    const float radius_1 = orbit.radius + sampler.normal_sample(radius_variance);

    const float orientation = orbit.orientation;
    const float orientation_velocity = sampler.normal_sample(params_.orientation_velocity_prior_variance);

    const Eigen::Vector3f center = orbit.center + sampler.normal_sample(state.plate_one().position_diagonal_covariance());
    const Eigen::Vector3f center_velocity = sampler.normal_sample(params_.center_velocity_prior_diagonal_covariance);

    return prediction(radius_0, radius_1, orientation, orientation_velocity, center, center_velocity);
  }

  __device__ void apply_process(
      const float& time_offset_seconds,
      util::default_rv_sampler& sampler,
      prediction& state) const noexcept {
    const float radius_noise_0 = sampler.normal_sample(params_.radius_process_variance);
    const float radius_noise_1 = sampler.normal_sample(params_.radius_process_variance);
    const float orientation_velocity_noise_0 = sampler.normal_sample(params_.orientation_velocity_process_variance);
    const float orientation_velocity_noise_1 = sampler.normal_sample(params_.orientation_velocity_process_variance);

    const Eigen::Vector3f center_velocity_noise_0 = sampler.normal_sample(params_.center_velocity_process_diagonal_covariance);
    const Eigen::Vector3f center_velocity_noise_1 = sampler.normal_sample(params_.center_velocity_process_diagonal_covariance);

    state.update_state(
        time_offset_seconds,
        radius_noise_0,
        radius_noise_1,
        orientation_velocity_noise_0,
        orientation_velocity_noise_1,
        center_velocity_noise_0,
        center_velocity_noise_1);
  }

  particle_filter_configuration(const particle_filter_configuration_parameters& params) noexcept : params_{params} {}
};

}  // namespace plate_orbit
