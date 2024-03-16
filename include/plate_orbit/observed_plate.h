#pragma once

#include <thrust/execution_policy.h>

#include <Eigen/Dense>

namespace plate_orbit {

class observed_plate {
 private:
  Eigen::Vector3f position_;
  Eigen::Vector3f position_diagonal_covariance_;

 public:
   [[nodiscard]] const Eigen::Vector3f& position() const noexcept { return position_; }

   [[nodiscard]] const Eigen::Vector3f& position_diagonal_covariance() const noexcept {
    return position_diagonal_covariance_;
  }

  
  observed_plate(const Eigen::Vector3f& position, const Eigen::Vector3f& position_diagonal_covariance) noexcept
      : position_{position}, position_diagonal_covariance_{position_diagonal_covariance} {}
};

}  // namespace plate_orbit
