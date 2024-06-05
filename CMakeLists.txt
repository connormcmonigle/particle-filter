cmake_minimum_required (VERSION 3.24)
project (particle_filter LANGUAGES CXX CUDA)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

include (GNUInstallDirs)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -flto")

add_subdirectory (pybind11)

pybind11_add_module (
  particle_filter
  src/particle_filter/particle_filter.cc
  
  src/point/init.cc
  src/point/particle_filter.cu

  src/plate_orbit/init.cc
  src/plate_orbit/particle_filter.cu

  src/fast_plate_orbit/init.cc
  src/fast_plate_orbit/particle_filter.cu
)

target_compile_features (particle_filter PRIVATE cxx_std_17)

target_include_directories (particle_filter PRIVATE include eigen ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(particle_filter PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(particle_filter PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --expt-relaxed-constexpr
                       --extended-lambda
                       --use_fast_math
                       >)

install (TARGETS particle_filter DESTINATION .)
