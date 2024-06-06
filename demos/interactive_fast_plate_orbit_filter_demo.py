from dataclasses import dataclass
import math
import time
import cv2
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, List
from threading import Lock
from particle_filter import fast_plate_orbit


ANGULAR_VELOCITY_STEP_SIZE = 2.0
POSITION_DIAGONAL_COVARIANCE = np.array([0.0001, 0.0001, 0.0001])
RADIAL_OFFSETS = [0.0, 0.0, 0.0, 0.0]
Z_OFFSETS = [0.0, 0.007, 0.0, 0.007]
VISIBILITY_ANGLE_THRESHOLD = math.pi / 3.0


@dataclass
class SimulatedRobotState:
    radius: float
    angle: float
    angular_velocity: float
    center: npt.NDArray[np.float64]
    observer: npt.NDArray[np.float64]

    def plate_positions(self) -> List[npt.NDArray[np.float64]]:
        offsets = [(0) * math.pi, (1 / 2) * math.pi,
                   (1) * math.pi, (3 / 2) * math.pi]

        def to_plate(offset, radial_offset, z_offset):
            horizontal_displacement = np.array(
                [math.cos(self.angle + offset), math.sin(self.angle + offset), 0.0])
            vertical_displacement = np.array([0.0, 0.0, z_offset])
            return self.center + (self.radius + radial_offset) * horizontal_displacement + vertical_displacement

        return [to_plate(*args) for args in zip(offsets, RADIAL_OFFSETS, Z_OFFSETS)]

    def visible_plate_positions(self) -> List[npt.NDArray[np.float64]]:
        visible_plates = []
        to_observer = self.observer - self.center
        for plate in self.plate_positions():
            to_plate = plate - self.center
            similarity = np.dot(to_plate, to_observer) / \
                (np.linalg.norm(to_observer) * np.linalg.norm(to_plate))
            if similarity >= math.cos(VISIBILITY_ANGLE_THRESHOLD):
                visible_plates.append(plate)
        return visible_plates

    def step_up_velocity(self):
        self.angular_velocity += ANGULAR_VELOCITY_STEP_SIZE

    def step_down_velocity(self):
        self.angular_velocity -= ANGULAR_VELOCITY_STEP_SIZE

    def update(self, offset_seconds: float):
        self.angle += offset_seconds * self.angular_velocity

    def set_center(self, center: npt.NDArray[np.float64]):
        self.center = center

    def set_observer(self, observer: npt.NDArray[np.float64]):
        self.observer = observer


class InteractivePointFilterDemo:
    window_title: str
    h: int
    w: int
    simulated_robot_state: SimulatedRobotState
    number_of_particles: int
    particle_filter_configuration: fast_plate_orbit.ParticleFilterConfigurationParameters
    input_image: npt.NDArray[np.uint8]
    filter_lock: Lock
    latest_update_time: Optional[float]
    filter: Optional[fast_plate_orbit.ParticleFilter]

    def __init__(self, window_title: str, resolution: Tuple[int, int], simulated_robot_state: SimulatedRobotState, number_of_particles: int, config: fast_plate_orbit.ParticleFilterConfigurationParameters):
        self.window_title = window_title
        self.h, self.w = resolution
        self.simulated_robot_state = simulated_robot_state
        self.number_of_particles = number_of_particles
        self.particle_filter_configuration = config
        self.input_image = np.zeros((self.h, self.w, 3), np.uint8)
        self.state_lock = Lock()
        self.latest_update_time = None
        self.filter = None
        self.reset = False
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, self.input_image)
        cv2.setMouseCallback(window_title, self.on_mouse)

    def _get_observation(self) -> fast_plate_orbit.Observation:
        plates = self.simulated_robot_state.visible_plate_positions()
        assert len(plates) == 1 or len(plates) == 2
        if len(plates) == 1:
            plate_one, = plates
            observation = fast_plate_orbit.Observation.from_one_plate(
                self.simulated_robot_state.observer,
                fast_plate_orbit.ObservedPlate(
                    plate_one, POSITION_DIAGONAL_COVARIANCE),
            )
        elif len(plates) == 2:
            plate_one, plate_two = plates
            observation = fast_plate_orbit.Observation.from_two_plates(
                self.simulated_robot_state.observer,
                fast_plate_orbit.ObservedPlate(
                    plate_one, POSITION_DIAGONAL_COVARIANCE),
                fast_plate_orbit.ObservedPlate(
                    plate_two, POSITION_DIAGONAL_COVARIANCE),
            )
        return observation

    def initalize_filter(self):
        observation = self._get_observation()
        self.filter = fast_plate_orbit.ParticleFilter(
            self.number_of_particles, observation, self.particle_filter_configuration)
        self.latest_update_time = time.time()

    def update_filter(self) -> fast_plate_orbit.Prediction:
        current_time = time.time()
        elapsed = current_time - self.latest_update_time
        self.simulated_robot_state.update(elapsed)
        observation = self._get_observation()
        self.filter.update_state_with_observation(elapsed, observation)
        self.latest_update_time = current_time
        return self.filter.extrapolate_state(0.0)

    def on_mouse(self, event, x: int, y: int, flags, param):
        x_value, y_value = x / self.w, y / self.h
        position = np.array([x_value, y_value, 0.0])

        with self.state_lock:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.simulated_robot_state.step_up_velocity()
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.simulated_robot_state.step_down_velocity()

            if self.filter is not None and self.latest_update_time is not None:
                self.simulated_robot_state.set_center(position)
                _ = self.update_filter()
            else:
                self.simulated_robot_state.set_center(position)
                self.initalize_filter()

    def render_loop(self):
        RED = (0, 0, 255)
        BLUE = (255, 0, 0)
        GREEN = (0, 255, 0)
        WHITE = (255, 255, 255)
        CYAN = (255, 255, 0)
        SPACE_KEY = 32

        while True:
            if self.filter is not None and self.latest_update_time is not None:
                with self.state_lock:
                    prediction = self.update_filter()

                self.input_image = np.zeros((self.h, self.w, 3), np.uint8)

                for plate in prediction.predicted_plates():
                    x, y, _ = plate.position()
                    v_x, v_y, _ = plate.velocity()
                    center = int(self.w * x), int(self.h * y)
                    end = int(self.w * (x + 0.5 * v_x)
                              ), int(self.h * (y + 0.5 * v_y))
                    cv2.arrowedLine(self.input_image, center,
                                    end, color=GREEN, thickness=5)
                    cv2.circle(self.input_image, center,
                               radius=2, color=GREEN, thickness=1)
                
                x, y, z = prediction.center()                
                v_x, v_y = prediction.center_velocity()
                center = int(self.w * x), int(self.h * y)
                end = int(self.w * (x + 0.5 * v_x)
                          ), int(self.h * (y + 0.5 * v_y))
                radii = int(self.w * prediction.radius()), int(self.h * prediction.radius())
                
                degrees = (180.0 / math.pi) * prediction.orientation()
                cv2.ellipse(self.input_image, center, radii, degrees,
                            0.0, 360.0, color=WHITE, thickness=3)
                cv2.circle(self.input_image, center, radius=8,
                           color=WHITE, thickness=-1)
                cv2.arrowedLine(self.input_image, center,
                                end, color=WHITE, thickness=5)

                for plate in self.simulated_robot_state.plate_positions():
                    x, y, _ = plate
                    center = int(self.w * x), int(self.h * y)
                    cv2.circle(self.input_image, center,
                               radius=8, color=RED, thickness=-1)

                for plate in self.simulated_robot_state.visible_plate_positions():
                    x, y, _ = plate
                    center = int(self.w * x), int(self.h * y)
                    cv2.circle(self.input_image, center,
                               radius=8, color=BLUE, thickness=-1)

                x, y, _ = self.simulated_robot_state.observer
                center = int(self.w * x), int(self.h * y)
                cv2.circle(self.input_image, center,
                           radius=100, color=CYAN, thickness=-1)

            cv2.imshow(self.window_title, self.input_image)

            key = cv2.waitKey(16)
            if key == SPACE_KEY:
                self.input_image = np.zeros((self.h, self.w, 3), np.uint8)

                self.input_image[:, :] = np.array(RED)
                cv2.imshow(self.window_title, self.input_image)
                _ = cv2.waitKey(200)

                with self.state_lock:
                    self.initalize_filter()
                _ = cv2.waitKey(16)


if __name__ == "__main__":
    config = fast_plate_orbit.ParticleFilterConfigurationParameters(
        0.11, 2.0, 0.001, 0.005, 0.0001, 6.0, 1.5, 3.0, np.array([6.0, 6.0]), np.array([10.0, 10.0]))
    simulated_robot_state = SimulatedRobotState(
        radius=0.1, angle=0.0, angular_velocity=0.0, center=np.zeros(3), observer=np.zeros(3))
    demo = InteractivePointFilterDemo(window_title="demo", resolution=(
        1024, 1024), simulated_robot_state=simulated_robot_state, number_of_particles=(1 << 20), config=config)
    demo.render_loop()
