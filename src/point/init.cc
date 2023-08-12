// point
#include <point/observation.h>
#include <point/particle_filter.h>
#include <point/particle_filter_configuration_parameters.h>
#include <point/prediction.h>

// pybind11
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace point {

namespace py = pybind11;

void init(py::module_& m) noexcept {
  auto point = m.def_submodule("point");

  py::class_<observation>(point, "Observation")
      .def(py::init<Eigen::Vector3f, Eigen::Vector3f>())
      .def("position", &observation::position)
      .def("position_diagonal_covariance", &observation::position_diagonal_covariance);

  py::class_<prediction>(point, "Prediction")
      .def(py::init<Eigen::Vector3f, Eigen::Vector3f>())
      .def("position", &prediction::position)
      .def("velocity", &prediction::velocity)
      .def("extrapolate_state", &prediction::extrapolate_state);

  py::class_<particle_filter_configuration_parameters>(point, "ParticleFilterConfigurationParameters")
      .def(py::init<Eigen::Vector3f, Eigen::Vector3f>());

  py::class_<particle_filter>(point, "ParticleFilter")
      .def(py::init<size_t, observation, particle_filter_configuration_parameters>())
      .def("extrapolate_state", &particle_filter::extrapolate_state, py::call_guard<py::gil_scoped_release>())

      .def(
          "update_state_sans_observation",
          &particle_filter::update_state_sans_observation,
          py::call_guard<py::gil_scoped_release>())

      .def(
          "update_state_with_observation",
          &particle_filter::update_state_with_observation,
          py::call_guard<py::gil_scoped_release>());
}

}  // namespace point
