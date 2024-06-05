#pragma once

#include <fast_plate_orbit/observed_plate.h>
#include <thrust/optional.h>

#include <Eigen/Dense>

namespace fast_plate_orbit {

class observation {
 private:
  Eigen::Vector3f observer_position_;
  observed_plate plate_one_;
  thrust::optional<observed_plate> plate_two_;

 public:
  __host__ __device__ [[nodiscard]] const Eigen::Vector3f& observer_position() const noexcept { return observer_position_; }
  __host__ __device__ [[nodiscard]] const observed_plate& plate_one() const noexcept { return plate_one_; }
  __host__ __device__ [[nodiscard]] const thrust::optional<observed_plate>& plate_two() const noexcept { return plate_two_; }

  __host__ __device__ observation(const Eigen::Vector3f& observer_position, const observed_plate& plate_one) noexcept
      : observer_position_{observer_position}, plate_one_{plate_one}, plate_two_{thrust::nullopt} {}

  __host__ __device__
  observation(const Eigen::Vector3f& observer_position, const observed_plate& plate_one, const observed_plate& plate_two) noexcept
      : observer_position_{observer_position}, plate_one_{plate_one}, plate_two_{plate_two} {}

  __host__ __device__ static observation from_one_plate(
      const Eigen::Vector3f& observer_position,
      const observed_plate& plate_one) noexcept {
    return observation(observer_position, plate_one);
  }

  __host__ __device__ static observation from_two_plates(
      const Eigen::Vector3f& observer_position,
      const observed_plate& plate_one,
      const observed_plate& plate_two) noexcept {
    return observation(observer_position, plate_one, plate_two);
  }
};

}  // namespace plate_orbit
