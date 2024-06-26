from typing import List
import numpy as np
import numpy.typing as npt


class ObservedPlate:
    def __init__(self, position: npt.NDArray[np.float64],
                 position_diagonal_covariance: npt.NDArray[np.float64]): ...

    def position(self) -> npt.NDArray[np.float64]: ...
    def position_diagonal_covariance(self) -> npt.NDArray[np.float64]: ...


class PredictedPlate:
    def __init__(
        self, position: npt.NDArray[np.float64], velocity: npt.NDArray[np.float64]): ...

    def position(self) -> npt.NDArray[np.float64]: ...
    def velocity(self) -> npt.NDArray[np.float64]: ...


class Observation:
    @staticmethod
    def from_one_plate(observer_position: npt.NDArray[np.float64],
                       plate_one: ObservedPlate) -> Observation: ...

    @staticmethod
    def from_two_plates(observer_position: npt.NDArray[np.float64], plate_one: ObservedPlate,
                        plate_two: ObservedPlate) -> Observation: ...


class Prediction:
    def predicted_plates(self) -> List[PredictedPlate]: ...
    def radius(self) -> float: ...
    def orientation(self) -> float: ...
    def orientation_velocity(self) -> float: ...
    def center(self) -> npt.NDArray[np.float64]: ...
    def center_velocity(self) -> npt.NDArray[np.float64]: ...
    def extrapolate_state(
        self, time_offset_seconds: float) -> Prediction: ...


class ParticleFilterConfigurationParameters:
    def __init__(
        self,
        radius_prior: float,
        similarity_logit_coefficient: float,
        radius_prior_variance_one_plate: float,
        radius_prior_variance_two_plates: float,
        radius_process_variance: float,
        orientation_velocity_prior_variance: float,
        orientation_velocity_process_variance: float,
        center_z_position_process_variance: float,
        center_xy_velocity_prior_diagonal_covariance: npt.NDArray[np.float64],
        center_xy_velocity_process_diagonal_covariance: npt.NDArray[np.float64]
    ): ...


class ParticleFilter:
    def __init__(self, number_of_samples: int, initial_observation: Observation,
                 params: ParticleFilterConfigurationParameters): ...

    def extrapolate_state(
        self, time_offset_seconds: float) -> Prediction: ...

    def update_state_sans_observation(self, time_offset_seconds: float): ...

    def update_state_with_observation(
        self, time_offset_seconds: float, state: Observation): ...
