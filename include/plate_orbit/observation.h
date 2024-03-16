#pragma once

#include <plate_orbit/observed_plate.h>
#include <thrust/optional.h>

#include <Eigen/Dense>

namespace plate_orbit {

class observation {
 private:
  Eigen::Vector3f observer_position_;
  observed_plate plate_one_;
  thrust::optional<observed_plate> plate_two_;

 public:
   [[nodiscard]] const Eigen::Vector3f& observer_position() const noexcept { return observer_position_; }
   [[nodiscard]] const observed_plate& plate_one() const noexcept { return plate_one_; }
   [[nodiscard]] const thrust::optional<observed_plate>& plate_two() const noexcept { return plate_two_; }

   observation(const Eigen::Vector3f& observer_position, const observed_plate& plate_one) noexcept
      : observer_position_{observer_position}, plate_one_{plate_one}, plate_two_{thrust::nullopt} {}

  
  observation(const Eigen::Vector3f& observer_position, const observed_plate& plate_one, const observed_plate& plate_two) noexcept
      : observer_position_{observer_position}, plate_one_{plate_one}, plate_two_{plate_two} {}

   static observation from_one_plate(
      const Eigen::Vector3f& observer_position,
      const observed_plate& plate_one) noexcept {
    return observation(observer_position, plate_one);
  }

   static observation from_two_plates(
      const Eigen::Vector3f& observer_position,
      const observed_plate& plate_one,
      const observed_plate& plate_two) noexcept {
    return observation(observer_position, plate_one, plate_two);
  }
};

}  // namespace plate_orbit
