# Introduction

This project provides a generic header only CUDA library enabling users to create custom GPU accelerated SIR particle filters. The library relies on the Thrust parallel algorithms library and supports both CUDA and OMP Thrust backends, enabling the library to be used even if a user does not have a CUDA capable GPU, albeit with significantly reduced throughput.

There are many real-time nonlinear state estimation applications for which Kalman Filters (KFs) and Extended Kalman Filters (EKFs) are inadequate. These applications include those for which the respective latent state space distributions have multiple modes as well as applications for which the respective conditional measurement likelihood distributions are non-guassian. By comparison, particle filters can represent arbitrary latent space distributions and impose no limitations on one's choice of conditional measurement likelihood distribution. This flexibility comes with a hefty computational cost which has traditionally largely excluded particle filters from consideration in real-time nonlinear state estimation applications. This project seeks to address the challenges posed by this computational cost with a work efficient GPU accelerated particle filter implementation which enables millions of particles to be processed in real-time. A key component in the provided GPU accelerated particle filter implementation is a novel parallel systematic resampling algorithm which does not suffer from thread divergence even when processing pessimal/extremely degenerate particle distributions.

# Example

For testing purposes and to serve as a reference implementation when adding custom particle filters, an example particle filter is included with the test suite; namely, a first order linear motion model `point` particle filter. This reference point particle filter implementation can be found [here](test/include/point/)


<div align="center">
<img src=".github/point.gif" width="320" height="320" />
</div>

# Consuming as a Header-only CUDA Library

Given `nvcc` lacks link time optimization support, this project is header-only for performance reasons. The intent is that the library can be easily consumed as a header-only CUDA library targeting c++20. When integrating with an existing c++ project, it will almost always be desirable to isolate the rest of your project from CUDA by way of wrapping template instantiations of the provided [`pf::filter::particle_filter`](pf/filter/particle_filter.h) template class using the PImpl idiom. See [this](#3-implement-pimpl-wrapper) step for more specifics regarding wrapping template instantions of the [`pf::filter::particle_filter`](pf/filter/particle_filter.h) template class using the PImpl idiom.

# Adding Custom Particle Filters

Outlined below are the steps required to add a custom particle filter using this library. The provided [`point`](/test/include/point) particle filter should serve as reference throughout this process.

At a high level, this project provides a [`pf::filter::particle_filter`](pf/filter/particle_filter.h) template class which accepts a user provided `ParticleFilterConfiguration` template type argument. The user provided `ParticleFilterConfiguration` type must define a prediction/hidden state type and an observation/measurement type. Additionally, the user provided `ParticleFilterConfiguration` type must define how to compute the conditional log-likelihood of a measurement given a particle, apply a hidden state transition to a particle, sample particles given a measurement (used for initialization) and reduce particles to obtain an overall prediction. The [`pf::filter::particle_filter`](pf/filter/particle_filter.h) template class relies on the definitions provided by the user's `ParticleFilterConfiguration` template type argument to implement optimized predict, update and resample steps for the user's desired application.


## 1. Define `observation` and `prediction`

Your chosen `observation` and `prediction` types correspond to the measurement state and particle/hidden state for your application respectively. The only requirements to implement these concepts is that `prediction` defines an `extrapolate_state` member function which returns the hidden state estimated by the prediction after `t` seconds have elapsed and that `prediction` be default constructible. The following defines the exact constraints using concept syntax (see [here](pf/filter/concepts)):

```cpp
template <typename T>
concept observation = true;

template <typename T>
concept prediction = requires(const T p, const float t) {
  { T() } -> std::same_as<T>;
  { p.extrapolate_state(t) } -> std::same_as<T>;
};
```

Importantly, `prediction`'s default constructor must be decorated with `PF_TARGET_ATTRS` function. See [`test/include/point/observation.h`](test/include/point/observation.h) and [`include/point/prediction.h`](include/point/prediction.h) for reference observation and prediction implementations respectively.

## 2. Implement `particle_filter_configuration`

Next, you'll need to define the `particle_filter_configuration` type for your application which is used to instantiate the provided [`pf::filter::particle_filter`](pf/filter/particle_filter.h) template class. The `particle_filter_configuration` type must define the required member functions to instantiate the [`pf::filter::particle_filter`](pf/filter/particle_filter.h) template class. Specifically, you'll need to define how to compute the conditional log-likelihood of a measurement given a particle, apply your hidden state transition function to a particle, sample particles given a measurement (used for initialization) and reduce particles to obtain an overall prediction.

- `particle_filter_configuration::apply_process`: applies the hidden state transition function to a particle in-place.
- `particle_filter_configuration::sample_from`: samples particle states given a single measurement.
- `particle_filter_configuration::conditional_log_likelihood`: computes the log-likelihood of a measurement given a particle (i.e, assuming the particle corresponds the ground truth hidden state).
- `particle_filter_configuration::most_likely_particle_reduction`: provides a binary reduction over particles to obtain the most likely hidden state.

 The following defines the exact constraints outlined above using concept syntax (see [here](pf/filter/concepts)):

```cpp

template <typename T, typename P>
concept particle_reduction_operation =
    requires(const T op, const particle_reduction_state<P> a, const particle_reduction_state<P> b) {
      requires prediction<P>;
      { op(a, b) } -> std::same_as<particle_reduction_state<P>>;
    };


template <typename T>
concept particle_filter_configuration = requires(
    const T c,
    const float t,
    typename T::sampler_type s,
    const typename T::observation_type o,
    typename T::prediction_type p) {
  requires sampler<typename T::sampler_type>;
  requires observation<typename T::observation_type>;
  requires prediction<typename T::prediction_type>;

  { c.apply_process(t, s, p) } -> std::same_as<void>;
  { c.sample_from(s, o) } -> std::same_as<typename T::prediction_type>;
  { c.conditional_log_likelihood(std::as_const(s), o, p) } -> std::same_as<float>;
  { c.most_likely_particle_reduction() } -> particle_reduction_operation<typename T::prediction_type>;
};

```

Notably, the `particle_filter_configuration::apply_process`, `particle_filter_configuration::sample_from`, and `particle_filter_configuration::conditional_log_likelihood` member functions outlined above must be decorated with `PF_TARGET_ATTRS` functions. Additionally, your `particle_reduction_operation::operator()` must also be decorated likewise.

See [`test/include/point/particle_filter_configuration.h`](test/include/point/particle_filter_configuration.h) for a reference implementation.

## 3. Implement PImpl wrapper

Typically, you'll want to isolate the rest of your project from CUDA. Hence, wrapping the instantiated [`pf::filter::particle_filter`](pf/filter/particle_filter.h) template class using the PImpl pattern to decouple the implementation details from the object representation will likely be desirable. Assuming your `particle_filter_configuration`s constructor takes only `particle_filter_configuration_params` as an argument, the PImpl boilerplate should look something like the following:


```cpp
// particle_filter.h

class particle_filter {
 private:
  struct impl;

  struct impl_deleter {
    void operator()(impl* p_impl);
  };

  std::unique_ptr<impl, impl_deleter> p_impl_;

 public:
  [[nodiscard]] prediction extrapolate_state(const float& time_offset_seconds) const noexcept;

  void update_state_sans_observation(const float& time_offset_seconds) noexcept;
  void update_state_with_observation(const float& time_offset_seconds, const observation& observation_state) noexcept;

  particle_filter(
      const std::size_t& number_of_particles,
      const observation& initial_observation,
      const particle_filter_configuration_parameters& params) noexcept;
};

```

```cpp
// particle_filter.cu

struct particle_filter::impl : public pf::filter::particle_filter<particle_filter_configuration> {
 public:
  template <typename... Ts>
  impl(Ts&&... ts) : pf::filter::particle_filter<particle_filter_configuration>(std::forward<Ts>(ts)...) {}
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
```

# Tests

The provided test suite can be compile and run with the following commands:

```bash
git submodule update --init --recursive # only required if you're running unit tests.
mkdir build && cd build
cmake ..
make omp_tests # use `make cuda_tests` for cuda targets.
./omp_tests
```

Notably, the test suite uses the minimal, header only [boost ut](https://github.com/boost-ext/ut) testing framework. Currently test coverage consists of some miscellaneous small component tests as well as a full integration test of the end-to-end state estimation functionality, using the reference `point` particle filter implementation.