#pragma once

#if defined(PARTICLE_FILTER_TARGET_CUDA)

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/vector.h>

#define PARTICLE_FILTER_TARGET_ATTRS __host__ __device__

namespace target_config {

static constexpr auto& policy = thrust::cuda::par;

template <typename T>
using vector_type = thrust::cuda::vector<T>;

}  // namespace target_config

#elif defined(PARTICLE_FILTER_TARGET_OMP)

#define PARTICLE_FILTER_TARGET_ATTRS

#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/omp/vector.h>
#include <cmath>

namespace target_config {

static constexpr auto& policy = thrust::omp::par;

template <typename T>
using vector = thrust::omp::vector<T>;

}  // namespace target_config

#endif
