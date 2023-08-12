#include <plate_orbit/init.h>
#include <point/init.h>

// pybind11
#include <pybind11/pybind11.h>

PYBIND11_MODULE(particle_filter, m) {
    point::init(m);
    plate_orbit::init(m);
}
