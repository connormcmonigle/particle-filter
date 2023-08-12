#pragma once

#include <thrust/execution_policy.h>

#include <Eigen/Dense>

namespace point {

class prediction {
 private:
  Eigen::Vector3f position_;
  Eigen::Vector3f velocity_;

 public:
  __host__ __device__ void
  update_state(const float& time_offset_seconds, const Eigen::Vector3f& noise_0, const Eigen::Vector3f& noise_1) noexcept {
    static constexpr float one_half = 1.0 / 2.0;
    static constexpr float one_twelfth = 1.0 / 12.0;

    const float velocity_noise_scale = sqrtf(time_offset_seconds);
    const float position_noise_scale = sqrtf(one_twelfth) * powf(velocity_noise_scale, 3);

    const Eigen::Vector3f d_velocity = velocity_noise_scale * noise_1;
    const Eigen::Vector3f d_position =
        time_offset_seconds * velocity_ + one_half * time_offset_seconds * d_velocity + position_noise_scale * noise_0;

    position_ = position_ + d_position;
    velocity_ = velocity_ + d_velocity;
  }

  __host__ __device__ [[nodiscard]] prediction extrapolate_state(const float& time_offset_seconds) const noexcept {
    const Eigen::Vector3f extrapolated_position = position_ + time_offset_seconds * velocity_;
    return prediction(extrapolated_position, velocity_);
  }

  __host__ __device__ [[nodiscard]] const Eigen::Vector3f& position() const noexcept { return position_; }
  __host__ __device__ [[nodiscard]] const Eigen::Vector3f& velocity() const noexcept { return velocity_; }

  __host__ __device__ prediction() noexcept : position_{Eigen::Vector3f::Zero()}, velocity_{Eigen::Vector3f::Zero()} {}
  __host__ __device__ prediction(const Eigen::Vector3f& position, const Eigen::Vector3f& velocity) noexcept
      : position_{position}, velocity_{velocity} {}
};

}  // namespace point
