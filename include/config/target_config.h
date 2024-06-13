#pragma once

#if defined(PF_TARGET_CUDA)

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/vector.h>

#define PF_TARGET_ATTRS __host__ __device__
#define PF_TARGET_ONLY_ATTRS __device__

namespace target_config {

static constexpr auto& policy = thrust::cuda::par;

template <typename T>
using vector = thrust::cuda::vector<T>;

}  // namespace target_config

#elif defined(PF_TARGET_OMP)

#define PF_TARGET_ATTRS
#define PF_TARGET_ONLY_ATTRS

#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/omp/vector.h>
#include <cmath>

namespace target_config {

static constexpr auto& policy = thrust::omp::par;

template <typename T>
using vector = thrust::omp::vector<T>;

}  // namespace target_config

#endif
