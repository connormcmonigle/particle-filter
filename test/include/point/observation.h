#pragma once

#include <config/target_config.h>

#include <Eigen/Dense>

namespace point {

class observation {
 private:
  Eigen::Vector3f position_;
  Eigen::Vector3f position_diagonal_covariance_;

 public:
  PARTICLE_FILTER_TARGET_ATTRS [[nodiscard]] const Eigen::Vector3f& position() const noexcept { return position_; }

  PARTICLE_FILTER_TARGET_ATTRS [[nodiscard]] const Eigen::Vector3f& position_diagonal_covariance() const noexcept {
    return position_diagonal_covariance_;
  }

  PARTICLE_FILTER_TARGET_ATTRS observation(const Eigen::Vector3f& position, const Eigen::Vector3f& position_diagonal_covariance) noexcept
      : position_{position}, position_diagonal_covariance_{position_diagonal_covariance} {}
};

}  // namespace point
