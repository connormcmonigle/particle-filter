cmake_minimum_required (VERSION 3.24)
project (particle_filter LANGUAGES CXX CUDA)

include (GNUInstallDirs)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffast-math -Wpedantic -Wall -Wextra -flto -fopenmp -lgomp")

add_executable(omp_tests test/src/main.cc)
add_executable(cuda_tests test/src/main.cu)

target_compile_definitions(omp_tests PUBLIC PF_TARGET_OMP THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)
target_compile_definitions(cuda_tests PUBLIC PF_TARGET_CUDA THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)

target_compile_features (omp_tests PRIVATE cxx_std_20)
target_compile_features (cuda_tests PRIVATE cxx_std_20)

target_include_directories (omp_tests PRIVATE include test/include eigen ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories (cuda_tests PRIVATE include test/include eigen ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(cuda_tests PROPERTIES CUDA_ARCHITECTURES all-major)
target_compile_options(cuda_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --diag-suppress 2811
                       --expt-relaxed-constexpr
                       --extended-lambda
                       --use_fast_math
                       >)
