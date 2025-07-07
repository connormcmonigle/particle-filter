
#pragma once

#include <pf/filter/concepts/particle_filter_configuration.h>
#include <pf/filter/particle_filter.h>
#include <point/observation.h>
#include <point/particle_filter_configuration.h>
#include <point/particle_filter_configuration_parameters.h>
#include <point/prediction.h>

#include <Eigen/Dense>
#include <boost/ut.hpp>
#include <cstddef>
#include <iostream>
#include <vector>

boost::ut::suite<"particle_filter"> particle_filter_tests = [] {
  using namespace boost::ut;

  constexpr float dt = 0.15f;
  constexpr float precision = 0.05f;

  constexpr std::size_t number_of_particles = 32u << 20;
  constexpr std::size_t number_of_update_steps = 128u;

  const auto params = point::particle_filter_configuration_parameters{
      .velocity_prior_diagonal_covariance = (Eigen::Vector3f{} << 4.0f, 4.0f, 4.0f).finished(),
      .velocity_process_diagonal_covariance = (Eigen::Vector3f{} << 8.0f, 8.0f, 8.0f).finished(),
  };

  const auto observation_covariance = (Eigen::Vector3f{} << 0.01f, 0.01f, 0.01f).finished();
  const auto initial_position = (Eigen::Vector3f{} << 0.00f, 0.00f, 0.00f).finished();

  "predicts correct particle state"_test =
      [&](const Eigen::Vector3f& velocity) {
        auto ground_truth = point::prediction(initial_position, velocity);
        auto observation = point::observation(ground_truth.position(), observation_covariance);

        auto filter = pf::filter::particle_filter<point::particle_filter_configuration>(number_of_particles, observation, params);

        for (std::size_t i(0); i < number_of_update_steps; ++i) {
          ground_truth = ground_truth.extrapolate_state(dt);
          observation = point::observation(ground_truth.position(), observation_covariance);

          filter.update_state_with_observation(dt, observation);
        }

        const auto prediction_0 = filter.extrapolate_state(0.0f);

        expect(prediction_0.velocity().isApprox(ground_truth.velocity(), precision)) << prediction_0.velocity();
        expect(prediction_0.position().isApprox(ground_truth.position(), precision)) << prediction_0.position();

        ground_truth = ground_truth.extrapolate_state(dt);
        filter.update_state_sans_observation(dt);
        const auto prediction_1 = filter.extrapolate_state(0.0f);

        expect(prediction_1.velocity().isApprox(ground_truth.velocity(), precision)) << prediction_1.velocity();
        expect(prediction_1.position().isApprox(ground_truth.position(), precision)) << prediction_1.position();

        ground_truth = ground_truth.extrapolate_state(dt);
        const auto prediction_2 = filter.extrapolate_state(dt);

        expect(prediction_2.velocity().isApprox(ground_truth.velocity(), precision)) << prediction_2.velocity();
        expect(prediction_2.position().isApprox(ground_truth.position(), precision)) << prediction_2.position();
      } |
      std::vector{
          (Eigen::Vector3f{} << 0.5f, 0.5f, 0.5f).finished(),
          (Eigen::Vector3f{} << 1.0f, 1.0f, 1.0f).finished(),
          (Eigen::Vector3f{} << 2.0f, 2.0f, 2.0f).finished(),
          (Eigen::Vector3f{} << 2.0f, -3.0f, 0.0f).finished(),
          (Eigen::Vector3f{} << -5.0f, -5.0f, -5.0f).finished(),
      };
};
