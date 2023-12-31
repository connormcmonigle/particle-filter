import numpy as np
import numpy.typing as npt


class Observation:
    def __init__(self, position: npt.NDArray[np.float64],
                 position_diagonal_covariance: npt.NDArray[np.float64]): ...

    def position(self) -> npt.NDArray[np.float64]: ...
    def position_diagonal_covariance(self) -> npt.NDArray[np.float64]: ...


class Prediction:
    def __init__(
        self, position: npt.NDArray[np.float64], velocity: npt.NDArray[np.float64]): ...

    def position(self) -> npt.NDArray[np.float64]: ...
    def velocity(self) -> npt.NDArray[np.float64]: ...
    def extrapolate_state(
        self, time_offset_seconds: float) -> Prediction: ...


class ParticleFilterConfigurationParameters:
    def __init__(self, velocity_prior_diagonal_covariance:
                 npt.NDArray[np.float64], velocity_process_diagonal_covariance: npt.NDArray[np.float64]): ...


class ParticleFilter:
    def __init__(self, number_of_samples: int, initial_observation: Observation,
                 config: ParticleFilterConfigurationParameters): ...

    def extrapolate_state(
        self, time_offset_seconds: float) -> Prediction: ...

    def update_state_sans_observation(self, time_offset_seconds: float): ...

    def update_state_with_observation(
        self, time_offset_seconds: float, state: Observation): ...
