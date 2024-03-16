#pragma once

#include <plate_orbit/observed_plate.h>
#include <plate_orbit/observed_plate_orbit.h>
#include <thrust/execution_policy.h>

#include <Eigen/Dense>

namespace plate_orbit {

class observed_plate_orbit_builder {
 private:
  float radius_prior_;
  Eigen::Vector3f observer_position_;

 public:
   observed_plate_orbit from_one_plate(const observed_plate& plate_one) const noexcept {
    const Eigen::Vector3f observer_relative_position = plate_one.position() - observer_position_;
    const Eigen::Vector3f projected_relative{observer_relative_position[0], observer_relative_position[1], 0.0};
    const Eigen::Vector3f predicted_center = plate_one.position() + radius_prior_ * projected_relative.normalized();
    const Eigen::Vector3f center_relative_position = plate_one.position() - predicted_center;

    const float predicted_orientation = atan2(center_relative_position[1], center_relative_position[0]);

    return observed_plate_orbit{radius_prior_, predicted_orientation, predicted_center};
  }

   observed_plate_orbit
  from_two_plates(const observed_plate& plate_one, const observed_plate& plate_two) const noexcept {
    const Eigen::Vector3f midpoint = 0.5f * (plate_one.position() + plate_two.position());
    const Eigen::Vector3f one_to_two = plate_two.position() - plate_one.position();
    const Eigen::Vector3f delta = 0.5f * (Eigen::Vector3f{} << -one_to_two[1], one_to_two[0], 0.0f).finished();

    const Eigen::Vector3f predicted_center =
        thrust::max((midpoint + delta).eval(), (midpoint + -delta).eval(), [this](const auto& a, const auto& b) {
          return (a - observer_position_).norm() < (b - observer_position_).norm();
        });

    const float predicted_radius = M_SQRT2 * delta.norm();
    const Eigen::Vector3f center_relative_position = plate_one.position() - predicted_center;

    const float predicted_orientation = atan2(center_relative_position[1], center_relative_position[0]);

    return observed_plate_orbit{predicted_radius, predicted_orientation, predicted_center};
  }

   observed_plate_orbit_builder(const float& radius_prior, const Eigen::Vector3f& observer_position) noexcept
      : radius_prior_{radius_prior}, observer_position_{observer_position} {}
};

}  // namespace plate_orbit
