#include <fast_plate_orbit/particle_filter.h>
#include <fast_plate_orbit/particle_filter_configuration.h>
#include <filter/particle_filter.h>

#include <utility>

namespace fast_plate_orbit {

struct particle_filter::impl : public filter::particle_filter<particle_filter_configuration> {
 public:
  template <typename... Ts>
  impl(Ts&&... ts) : filter::particle_filter<particle_filter_configuration>(std::forward<Ts>(ts)...) {}
};

void particle_filter::impl_deleter::operator()(particle_filter::impl* p_impl) { delete p_impl; }

prediction particle_filter::extrapolate_state(const float& time_offset_seconds) const noexcept {
  return p_impl_->extrapolate_state(time_offset_seconds);
}

void particle_filter::update_state_sans_observation(const float& time_offset_seconds) noexcept {
  p_impl_->update_state_sans_observation(time_offset_seconds);
}

void particle_filter::update_state_with_observation(
    const float& time_offset_seconds,
    const observation& observation_state) noexcept {
  p_impl_->update_state_with_observation(time_offset_seconds, observation_state);
}

particle_filter::particle_filter(
    const std::size_t& number_of_particles,
    const observation& initial_observation,
    const particle_filter_configuration_parameters& params) noexcept
    : p_impl_(new particle_filter::impl(number_of_particles, initial_observation, params)) {}

}  // namespace fast_plate_orbit
