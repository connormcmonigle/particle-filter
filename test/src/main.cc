#include <filter/particle_filter.h>
#include <filter/concepts/particle_filter_configuration.h>
#include <point/observation.h>
#include <point/particle_filter_configuration.h>
#include <point/particle_filter_configuration_parameters.h>
#include <point/prediction.h>

#include <Eigen/Dense>
#include <cstddef>

int main() {
  const auto params = point::particle_filter_configuration_parameters{
      .velocity_prior_diagonal_covariance = (Eigen::Vector3f{} << 8.0, 8.0, 0.01).finished(),
      .velocity_process_diagonal_covariance = (Eigen::Vector3f{} << 8.0, 8.0, 0.01).finished(),
  };

  const auto pos = (Eigen::Vector3f{} << 1.0, 1.0, 0.0).finished();
  const auto cov = (Eigen::Vector3f{} << 0.1, 0.1, 0.1).finished();
  const auto initial = point::observation(pos, cov);

  constexpr std::size_t number_of_particles = static_cast<std::size_t>(1) << 20;
  auto filter = filter::particle_filter<point::particle_filter_configuration>(number_of_particles, initial, params);

  filter.update_state_with_observation(0.15f, initial);
}
