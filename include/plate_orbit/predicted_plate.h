#pragma once

#include <thrust/execution_policy.h>

#include <Eigen/Dense>

namespace plate_orbit {

class predicted_plate {
 private:
  Eigen::Vector3f position_;
  Eigen::Vector3f velocity_;

 public:
   [[nodiscard]] const Eigen::Vector3f& position() const noexcept { return position_; }
   [[nodiscard]] const Eigen::Vector3f& velocity() const noexcept { return velocity_; }

   predicted_plate() noexcept : position_{Eigen::Vector3f::Zero()}, velocity_{Eigen::Vector3f::Zero()} {}

   predicted_plate(const Eigen::Vector3f& position, const Eigen::Vector3f& velocity) noexcept
      : position_{position}, velocity_{velocity} {}
};

}  // namespace plate_orbit
