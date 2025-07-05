#pragma once

#include <pf/config/target_config.h>
#include <pf/filter/particle_reduction_state.h>

#include <boost/ut.hpp>

boost::ut::suite<"particle_reduction_state"> particle_reduction_state_tests = [] {
  using namespace boost::ut;

  "creates correct state from nothing"_test = [] {
    const auto state = pf::filter::particle_reduction_state<int>::zero();
    expect(0_i == state.most_likely_particle());
    expect(0_u == state.count());
  };

  "creates correct state from one particle"_test = [] {
    const auto state = pf::filter::particle_reduction_state<int>::from_one_particle(42);
    expect(42_i == state.most_likely_particle());
    expect(1_u == state.count());
  };

  "default constructor creates correct state"_test = [] {
    const auto state = pf::filter::particle_reduction_state<int>(42, 3u);
    expect(42_i == state.most_likely_particle());
    expect(3_u == state.count());
  };
};
