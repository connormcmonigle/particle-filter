#pragma once

#include <plate_orbit/observation.h>
#include <plate_orbit/predicted_plate.h>
#include <util/device_array.h>

#include <Eigen/Dense>
#include <array>

namespace plate_orbit {

namespace helper {

__device__ __host__ [[nodiscard]] inline float to_radius(const float& radius) noexcept {
  constexpr float min_radius = 0.01f;
  return thrust::max(min_radius, radius);
}

__device__ __host__ [[nodiscard]] inline float to_orientation(const float& angle_radians) noexcept {
  const float value = fmod(angle_radians, M_PI);
  return (value < 0.0f) ? value + M_PI : value;
}

}  // namespace helper

class prediction {
 private:
  static constexpr size_t number_of_plates = 4;

  float radius_0_;
  float radius_1_;

  float orientation_;
  float orientation_velocity_;

  Eigen::Vector3f center_;
  Eigen::Vector3f center_velocity_;

  struct angle_offset_and_radius {
    float angle_offset;
    float radius;
  };

 public:
  [[nodiscard]] std::array<predicted_plate, number_of_plates> predicted_plates_for_host() const noexcept {
    return predicted_plates().to_host_array();
  }

   [[nodiscard]] util::device_array<predicted_plate, number_of_plates> predicted_plates() const noexcept {
    const util::device_array<angle_offset_and_radius, number_of_plates> angle_offsets_and_radii = {
        angle_offset_and_radius{0.0, radius_0_},
        angle_offset_and_radius{M_PI_2, radius_1_},
        angle_offset_and_radius{M_PI, radius_0_},
        angle_offset_and_radius{M_PI + M_PI_2, radius_1_},
    };

    return angle_offsets_and_radii.transformed([this](const angle_offset_and_radius& value) {
      const float angle = value.angle_offset + orientation_;

      const Eigen::Vector3f predicted_plate_position =
          center_ + value.radius * (Eigen::Vector3f{} << cosf(angle), sinf(angle), 0.0f).finished();

      const Eigen::Vector3f predicted_plate_velocity =
          center_velocity_ +
          orientation_velocity_ * value.radius * (Eigen::Vector3f{} << -sinf(angle), cosf(angle), 0.0f).finished();

      return predicted_plate(predicted_plate_position, predicted_plate_velocity);
    });
  }

   [[nodiscard]] prediction extrapolate_state(const float& time_offset_seconds) const noexcept {
    return prediction(
        radius_0_,
        radius_1_,
        orientation_ + time_offset_seconds * orientation_velocity_,
        orientation_velocity_,
        center_ + time_offset_seconds * center_velocity_,
        center_velocity_);
  }

   void update_state(
      const float& time_offset_seconds,
      const float& radius_noise_0,
      const float& radius_noise_1,
      const float& orientation_velocity_noise_0,
      const float& orientation_velocity_noise_1,
      const Eigen::Vector3f& center_velocity_noise_0,
      const Eigen::Vector3f& center_velocity_noise_1) noexcept {
    static constexpr float one_half = 1.0 / 2.0;
    static constexpr float one_twelfth = 1.0 / 12.0;

    const float radius_noise_scale = sqrtf(time_offset_seconds);
    const float velocity_noise_scale = radius_noise_scale;
    const float position_noise_scale = sqrtf(one_twelfth) * powf(velocity_noise_scale, 3);

    const float d_radius_0 = radius_noise_scale * radius_noise_0;
    const float d_radius_1 = radius_noise_scale * radius_noise_1;

    const float d_orientation_velocity = velocity_noise_scale * orientation_velocity_noise_1;
    const float d_orientation = time_offset_seconds * orientation_velocity_ +
                                one_half * time_offset_seconds * d_orientation_velocity +
                                position_noise_scale * orientation_velocity_noise_0;

    const Eigen::Vector3f d_center_velocity = velocity_noise_scale * center_velocity_noise_1;
    const Eigen::Vector3f d_center = time_offset_seconds * center_velocity_ + one_half * time_offset_seconds * d_center_velocity +
                                     position_noise_scale * center_velocity_noise_0;

    radius_0_ = helper::to_radius(radius_0_ + d_radius_0);
    radius_1_ = helper::to_radius(radius_1_ + d_radius_1);

    orientation_ = helper::to_orientation(orientation_ + d_orientation);
    orientation_velocity_ = orientation_velocity_ + d_orientation_velocity;

    center_ = center_ + d_center;
    center_velocity_ = center_velocity_ + d_center_velocity;
  }

   [[nodiscard]] const float& radius_0() const noexcept { return radius_0_; }
   [[nodiscard]] const float& radius_1() const noexcept { return radius_1_; }
   [[nodiscard]] const float& orientation() const noexcept { return orientation_; }
   [[nodiscard]] const float& orientation_velocity() const noexcept { return orientation_velocity_; }
   [[nodiscard]] const Eigen::Vector3f& center() const noexcept { return center_; }
   [[nodiscard]] const Eigen::Vector3f& center_velocity() const noexcept { return center_velocity_; }

   prediction() noexcept
      : radius_0_{0.0f},
        radius_1_{0.0f},
        orientation_{0.0f},
        orientation_velocity_{0.0f},
        center_{Eigen::Vector3f::Zero()},
        center_velocity_{Eigen::Vector3f::Zero()} {}

   prediction(
      const float& radius_0,
      const float& radius_1,
      const float& orientation,
      const float& orientation_velocity,
      const Eigen::Vector3f& center,
      const Eigen::Vector3f& center_velocity) noexcept
      : radius_0_{helper::to_radius(radius_0)},
        radius_1_{helper::to_radius(radius_1)},
        orientation_{helper::to_orientation(orientation)},
        orientation_velocity_{orientation_velocity},
        center_{center},
        center_velocity_{center_velocity} {}
};

}  // namespace plate_orbit
