cmake_minimum_required (VERSION 3.24)
project (particle_filter LANGUAGES CXX CUDA)
include (GNUInstallDirs)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffast-math -Wpedantic -Wall -Wextra -flto -fopenmp -lgomp")

add_executable(tests test/src/main.cc)

add_definitions(-DPF_TARGET_OMP)
target_compile_features (tests PRIVATE cxx_std_20)
target_include_directories (tests PRIVATE include test/include eigen ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(tests PROPERTIES CUDA_ARCHITECTURES all-major)
target_compile_options(tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --expt-relaxed-constexpr
                       --extended-lambda
                       --use_fast_math
                       >)
