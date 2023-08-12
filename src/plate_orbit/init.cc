// plate orbit
#include <plate_orbit/observation.h>
#include <plate_orbit/observed_plate.h>
#include <plate_orbit/particle_filter.h>
#include <plate_orbit/particle_filter_configuration_parameters.h>
#include <plate_orbit/predicted_plate.h>
#include <plate_orbit/prediction.h>

// pybind11
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace plate_orbit {

namespace py = pybind11;

void init(py::module_& m) noexcept {
  auto plate_orbit = m.def_submodule("plate_orbit");

  py::class_<observed_plate>(plate_orbit, "ObservedPlate")
      .def(py::init<Eigen::Vector3f, Eigen::Vector3f>())
      .def("position", &observed_plate::position)
      .def("position_diagonal_covariance", &observed_plate::position_diagonal_covariance);

  py::class_<predicted_plate>(plate_orbit, "PredictedPlate")
      .def(py::init<Eigen::Vector3f, Eigen::Vector3f>())
      .def("position", &predicted_plate::position)
      .def("velocity", &predicted_plate::velocity);

  py::class_<observation>(plate_orbit, "Observation")
      .def_static("from_one_plate", &observation::from_one_plate)
      .def_static("from_two_plates", &observation::from_two_plates);

  py::class_<prediction>(plate_orbit, "Prediction")
      .def("predicted_plates", &prediction::predicted_plates_for_host)
      .def("radius_0", &prediction::radius_0)
      .def("radius_1", &prediction::radius_1)
      .def("orientation", &prediction::orientation)
      .def("orientation_velocity", &prediction::orientation_velocity)
      .def("center", &prediction::center)
      .def("center_velocity", &prediction::center_velocity)
      .def("extrapolate_state", &prediction::extrapolate_state);

  py::class_<particle_filter_configuration_parameters>(plate_orbit, "ParticleFilterConfigurationParameters")
      .def(py::init<float, float, float, float, float, float, float, Eigen::Vector3f, Eigen::Vector3f>());

  py::class_<particle_filter>(plate_orbit, "ParticleFilter")
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

}  // namespace plate_orbit
