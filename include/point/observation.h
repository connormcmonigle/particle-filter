#pragma once

#include <thrust/execution_policy.h>

#include <Eigen/Dense>

namespace point {

class observation {
 private:
  Eigen::Vector3f position_;
  Eigen::Vector3f position_diagonal_covariance_;

 public:
  __host__ __device__ [[nodiscard]] const Eigen::Vector3f& position() const noexcept { return position_; }
  
  __host__ __device__ [[nodiscard]] const Eigen::Vector3f& position_diagonal_covariance() const noexcept {
    return position_diagonal_covariance_;
  }

  __host__ __device__ observation(const Eigen::Vector3f& position, const Eigen::Vector3f& position_diagonal_covariance) noexcept
      : position_{position}, position_diagonal_covariance_{position_diagonal_covariance} {}
};

}  // namespace point
