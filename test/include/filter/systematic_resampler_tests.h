#pragma once

#include <pf/config/target_config.h>
#include <pf/filter/systematic_resampler.h>

#include <algorithm>
#include <boost/ut.hpp>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

template <typename T, typename... Ts>
pf::target_config::vector<T> of(T t, Ts... ts) {
  const auto elements = {t, ts...};

  pf::target_config::vector<T> result{};
  std::copy(elements.begin(), elements.end(), std::back_inserter(result));

  return result;
}

boost::ut::suite<"systematic_resampler"> systematic_resampler_tests = [] {
  using namespace boost::ut;
  constexpr float n_inf = -std::numeric_limits<float>::infinity();

  "resamples correctly with one particle"_test = [] {
    pf::filter::systematic_resampler<int, std::size_t> resampler(1u);

    const pf::target_config::vector<float> log_weights = of<float>(0.0f);
    pf::target_config::vector<int> particles = of<int>(42);

    resampler.resample(log_weights, particles);

    expect(1_ul == particles.size());
    expect(42_i == *particles.begin());
  };

  "resamples correctly when all weight is on last particle"_test = [] {
    pf::filter::systematic_resampler<int, std::size_t> resampler(5u);

    const pf::target_config::vector<float> log_weights = of<float>(n_inf, n_inf, n_inf, n_inf, 0.0f);
    pf::target_config::vector<int> particles = of<int>(1, 3, 5, 7, 11);

    resampler.resample(log_weights, particles);

    expect(5_ul == particles.size());
    std::for_each(particles.begin(), particles.end(), [&](const auto& particle) { expect(11_i == particle); });
  };

  "resamples correctly when all weight is on first particle"_test = [] {
    pf::filter::systematic_resampler<int, std::size_t> resampler(5u);

    const pf::target_config::vector<float> log_weights = of<float>(0.0f, n_inf, n_inf, n_inf, n_inf);
    pf::target_config::vector<int> particles = of<int>(1, 3, 5, 7, 11);

    resampler.resample(log_weights, particles);

    expect(5_ul == particles.size());
    std::for_each(particles.begin(), particles.end(), [&](const auto& particle) { expect(1_i == particle); });
  };

  "resamples correctly when weights are uniformly distributed"_test = [] {
    pf::filter::systematic_resampler<int, std::size_t> resampler(5u);

    const pf::target_config::vector<float> log_weights = of<float>(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    pf::target_config::vector<int> particles = of<int>(1, 3, 5, 7, 11);

    resampler.resample(log_weights, particles);

    const pf::target_config::vector<int> expected = of<int>(1, 3, 5, 7, 11);
    expect(expected == particles);
  };

  "resamples correctly when weights are non-uniformly distributed"_test = [] {
    pf::filter::systematic_resampler<int, std::size_t> resampler(5u);

    const pf::target_config::vector<float> log_weights = of<float>(n_inf, n_inf, std::log(1.0f), std::log(2.0f), std::log(2.0f));
    pf::target_config::vector<int> particles = of<int>(1, 3, 5, 7, 11);

    resampler.resample(log_weights, particles);

    const pf::target_config::vector<int> expected = of<int>(5, 7, 7, 11, 11);
    expect(expected == particles);
  };
};
