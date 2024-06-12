#pragma once

#include <filter/concepts/observation.h>
#include <filter/concepts/prediction.h>
#include <filter/particle_reduction_state.h>

#include <concepts>

namespace filter {

namespace concepts {

template <typename T, typename P>
concept particle_reduction_operation =
    requires(const T op, const particle_reduction_state<P> a, const particle_reduction_state<P> b) {
      requires prediction<P>;
      { op(a, b) } -> std::same_as<particle_reduction_state<P>>;
    };

}  // namespace concepts

}  // namespace filter
