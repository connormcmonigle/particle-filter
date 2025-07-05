#pragma once

#include <Eigen/Dense>

namespace point {

struct particle_filter_configuration_parameters {
  Eigen::Vector3f velocity_prior_diagonal_covariance;
  Eigen::Vector3f velocity_process_diagonal_covariance;
};

}  // namespace point
