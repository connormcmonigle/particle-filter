#pragma once

#include <thrust/random.h>

#include <Eigen/Dense>
#include <cassert>
#include <random>

namespace util {

template <typename T, typename RandomNumberGenerator>
class random_variable_sampler {
 private:
  RandomNumberGenerator random_number_generator_{};
  thrust::normal_distribution<T> standard_normal_{};

 public:
  using floating_point_type = T;
  using rng_type = RandomNumberGenerator;

  __host__ __device__ void seed(const typename RandomNumberGenerator::result_type& seed) noexcept {
    random_number_generator_.seed(seed);
  }

  __host__ __device__ [[nodiscard]] T normal_sample(const T& variance) noexcept {
    return sqrt(variance) * standard_normal_(random_number_generator_);
  }

  template <int N>
  __host__ __device__ [[nodiscard]] Eigen::Matrix<T, N, 1> normal_sample(
      const Eigen::Matrix<T, N, 1>& diagonal_covariance) noexcept {
    return diagonal_covariance.cwiseSqrt().cwiseProduct(
        Eigen::Matrix<T, N, 1>{}.unaryExpr([this](auto) { return standard_normal_(random_number_generator_); }));
  }

  __host__ __device__ [[nodiscard]] T unnormalized_normal_log_density(const T& variance, const T& x) const noexcept {
    return -static_cast<T>(0.5) * (x * x) / variance;
  }

  template <int N>
  __host__ __device__ [[nodiscard]] T unnormalized_normal_log_density(
      const Eigen::Matrix<T, N, 1>& diagonal_covariance,
      const Eigen::Matrix<T, N, 1>& x) const noexcept {
    return -static_cast<T>(0.5) * x.cwiseProduct(diagonal_covariance.cwiseInverse()).dot(x);
  }
};

using default_rv_sampler = random_variable_sampler<float, thrust::random::default_random_engine>;

}  // namespace util
