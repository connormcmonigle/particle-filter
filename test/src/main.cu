#include <filter/particle_filter_tests.h>
#include <filter/particle_reduction_state_tests.h>
#include <filter/systematic_resampler_tests.h>

#include <boost/ut.hpp>

int main() { boost::ut::cfg<boost::ut::override> = {.tag = {"benchmark"}}; }
