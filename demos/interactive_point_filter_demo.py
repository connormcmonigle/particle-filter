import time
import cv2
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, List
from threading import Lock
from particle_filter import point

POSITION_DIAGONAL_COVARIANCE = np.array([0.0001, 0.0001, 0.0001])

class InteractivePointFilterDemo:
    window_title: str
    h: int
    w: int
    number_of_particles: int
    particle_filter_configuration: point.ParticleFilterConfigurationParameters
    input_image: npt.NDArray[np.uint8]
    filter_lock: Lock
    latest_update_time: Optional[float]
    x_value: float
    y_value: float
    filter: Optional[point.ParticleFilter]


    def __init__(self, window_title: str, resolution: Tuple[int, int], number_of_particles: int, config: point.ParticleFilterConfigurationParameters):
        self.window_title = window_title
        self.h, self.w = resolution
        self.number_of_particles = number_of_particles
        self.particle_filter_configuration = config
        self.input_image = np.zeros((self.h, self.w, 3), np.uint8)
        self.filter_lock = Lock()
        self.latest_update_time = None
        self.filter = None

        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, self.input_image)
        cv2.setMouseCallback(window_title, self.on_mouse)

    def _get_observation(self) -> point.Observation:
        position = np.array([self.x_value, self.y_value, 0.0])
        return point.Observation(position, POSITION_DIAGONAL_COVARIANCE)

    def update_filter(self) -> point.Prediction:
        current_time = time.time()
        elapsed = current_time - self.latest_update_time
        observation = self._get_observation()
        self.filter.update_state_with_observation(elapsed, observation)
        self.latest_update_time = current_time
        return self.filter.extrapolate_state(0.0)

    def initalize_filter(self):
        self.latest_update_time = time.time()
        observation = self._get_observation()
        self.filter = point.ParticleFilter(self.number_of_particles, observation, self.particle_filter_configuration)

    def on_mouse(self, event, x: int, y: int, flags, param):
        x_value, y_value = x / self.w, y / self.h
        self.x_value = x_value
        self.y_value = y_value

        with self.filter_lock:
            if self.filter is None or self.latest_update_time is None:
                self.initalize_filter()
            else:
                _ = self.update_filter()

    def render_loop(self):
        RED = (0, 0, 255)
        BLUE = (255, 0, 0)
        while True:
            if self.filter is not None and self.latest_update_time is not None:
                with self.filter_lock:
                    prediction = self.update_filter()

                self.input_image = np.zeros((self.h, self.w, 3), np.uint8)

                p_x, p_y, _ = prediction.position()
                v_x, v_y, _ = prediction.velocity()
                start = int(self.w * p_x), int(self.h * p_y)
                end = int(self.w * (p_x + 0.5 * v_x)), int(self.h * (p_y + 0.5 * v_y))

                cv2.arrowedLine(self.input_image, start, end, color=BLUE, thickness=5)
                cv2.circle(self.input_image, start, radius=16, color=RED, thickness=-1)

            cv2.imshow(self.window_title, self.input_image)
            cv2.waitKey(1)


if __name__ == "__main__":
    VELOCITY_PRIOR_DIAGONAL_COVARIANCE = np.array([8.0, 8.0, 0.01])
    VELOCITY_PROCESS_DIAGONAL_COVARIANCE = np.array([3.0, 3.0, 0.001])

    config = point.ParticleFilterConfigurationParameters(VELOCITY_PRIOR_DIAGONAL_COVARIANCE, VELOCITY_PROCESS_DIAGONAL_COVARIANCE)
    demo = InteractivePointFilterDemo(window_title="demo", resolution=(1024, 1024), number_of_particles=1 << 20, config=config)
    demo.render_loop()
