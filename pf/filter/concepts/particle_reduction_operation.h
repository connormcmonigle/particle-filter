#pragma once

#include <pf/filter/concepts/observation.h>
#include <pf/filter/concepts/prediction.h>
#include <pf/filter/particle_reduction_state.h>

#include <concepts>

namespace pf::filter::concepts {

template <typename T, typename P>
concept particle_reduction_operation =
    requires(const T op, const particle_reduction_state<P> a, const particle_reduction_state<P> b) {
      requires prediction<P>;
      { op(a, b) } -> std::same_as<particle_reduction_state<P>>;
    };

}  // namespace pf::filter::concepts
