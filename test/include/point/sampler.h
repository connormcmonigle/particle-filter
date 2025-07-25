#pragma once

#include <pf/config/target_config.h>
#include <thrust/random.h>

#include <Eigen/Dense>
#include <cassert>

namespace point {

class sampler {
 private:
  thrust::random::default_random_engine random_number_generator_{};
  thrust::normal_distribution<float> standard_normal_{};

 public:
  using random_number_generator_type = thrust::random::default_random_engine;

  PF_TARGET_ATTRS void seed(const thrust::random::default_random_engine::result_type& seed) noexcept {
    random_number_generator_.seed(seed);
  }

  PF_TARGET_ATTRS [[nodiscard]] thrust::random::default_random_engine& random_number_generator() noexcept {
    return random_number_generator_;
  }

  PF_TARGET_ATTRS [[nodiscard]] float normal_sample(const float& variance) noexcept {
    return sqrt(variance) * standard_normal_(random_number_generator_);
  }

  template <int N>
  PF_TARGET_ATTRS [[nodiscard]] Eigen::Matrix<float, N, 1> normal_sample(
      const Eigen::Matrix<float, N, 1>& diagonal_covariance) noexcept {
    return diagonal_covariance.cwiseSqrt().cwiseProduct(
        Eigen::Matrix<float, N, 1>{}.unaryExpr([this](auto) { return standard_normal_(random_number_generator_); }));
  }

  PF_TARGET_ATTRS [[nodiscard]] float unnormalized_normal_log_density(const float& variance, const float& x) const noexcept {
    return -0.5f * (x * x) / variance;
  }

  template <int N>
  PF_TARGET_ATTRS [[nodiscard]] float unnormalized_normal_log_density(
      const Eigen::Matrix<float, N, 1>& diagonal_covariance,
      const Eigen::Matrix<float, N, 1>& x) const noexcept {
    return -0.5f * x.cwiseProduct(diagonal_covariance.cwiseInverse()).dot(x);
  }
};

}  // namespace point
