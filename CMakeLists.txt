cmake_minimum_required (VERSION 3.24)
project (particle_filter LANGUAGES CXX)

include (GNUInstallDirs)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffast-math -Wall -Wextra -flto -fopenmp -lgomp")

add_subdirectory (pybind11)

pybind11_add_module (
  particle_filter
  src/particle_filter/particle_filter.cc
  
  src/point/init.cc
  src/point/particle_filter.cc

  src/plate_orbit/init.cc
  src/plate_orbit/particle_filter.cc
)

target_compile_features (particle_filter PRIVATE cxx_std_17)
target_include_directories (particle_filter PRIVATE include eigen $ENV{THRUST_ROOT})

install (TARGETS particle_filter DESTINATION .)
