import os
import time

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from smartredis import Client


class SimpleClimateBiasCorrectionEnv(gym.Env):
    """
    A gym environment for a simple climate bias correction problem,
    using a Fortran model for temperature evolution and calculating the cost (reward) in Python.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, seed, render_mode=None):
        """
        Initialize the environment, action space, and observation space.
        """
        self.min_temperature = 0.0
        self.max_temperature = 1.0
        self.max_heating_rate = 1.0
        self.dt = 1.0  # Time step (minutes)
        self.count = 0.0
        self.screen = None
        self.clock = None
        self.seed = seed

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-self.max_heating_rate,
            high=self.max_heating_rate,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([self.min_temperature], dtype=np.float32),
            high=np.array([self.max_temperature], dtype=np.float32),
            dtype=np.float32,
        )

        assert (
            render_mode is None or render_mode in self.metadata["render_modes"]
        )
        self.render_mode = render_mode

        # SmartRedis setup
        self.REDIS_ADDRESS = os.getenv("SSDB")
        if self.REDIS_ADDRESS is None:
            raise EnvironmentError("SSDB environment variable is not set.")
        self.redis = Client(address=self.REDIS_ADDRESS, cluster=False)
        print(f"[RL Env] Connected to Redis server: {self.REDIS_ADDRESS}")

        self.reset(self.seed)

    def step(self, u):
        """
        Performs one step in the environment using the action `u` (heating increment).

        Args:
            u (float): The action, representing a heating increment.

        Returns:
            tuple: A tuple containing the new observation, the reward, whether the episode is done,
                   and additional information.
        """

        # Clip action to the allowed range
        u = np.clip(u, -self.max_heating_rate, self.max_heating_rate)[0]

        # Send the heating increment to Redis
        self.redis.put_tensor(
            f"py2f_redis_s{self.seed}", np.array([u], dtype=np.float64)
        )
        self.redis.put_tensor(
            f"SIGCOMPUTE_S{self.seed}", np.array([1], dtype=np.int32)
        )

        # Wait for the Fortran model to compute the new temperature and retrieve it
        new_temperature = None
        while new_temperature is None:
            if self.redis.tensor_exists(f"f2py_redis_s{self.seed}"):
                new_temperature = self.redis.get_tensor(
                    f"f2py_redis_s{self.seed}"
                )[0]
                time.sleep(0.01)
                self.redis.delete_tensor(f"f2py_redis_s{self.seed}")
            else:
                continue  # Wait for the computation to complete

        # Clip the new temperature to the allowed range
        new_temperature = np.clip(
            new_temperature, self.min_temperature, self.max_temperature
        )

        # Update the state
        self.state = np.array([new_temperature])

        # Calculate the cost (mean squared error) in Python
        observed_temperature = (321.75 - 273.15) / 100
        costs = (observed_temperature - new_temperature) ** 2

        # Return the new observation, negative cost as reward, and False for 'done' (no termination condition)
        return self._get_obs(), -costs, False, False, self._get_info()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.

        Returns:
            np.array: The initial observation.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed

        # Send the start signal to the Fortran model
        self.redis.put_tensor(
            f"SIGSTART_S{self.seed}", np.array([1], dtype=np.int32)
        )

        # Wait for the Fortran model to compute the new temperature and retrieve it
        initial_temperature = None
        while initial_temperature is None:
            if self.redis.tensor_exists(f"f2py_redis_s{self.seed}"):
                initial_temperature = self.redis.get_tensor(
                    f"f2py_redis_s{self.seed}"
                )[0]
                time.sleep(0.25)
                self.redis.delete_tensor(f"f2py_redis_s{self.seed}")
            else:
                continue  # Wait for the computation to complete

        self.state = np.array([initial_temperature])

        return self._get_obs(), self._get_info()

    def _get_info(self):
        return {"_": None}

    def _get_obs(self):
        """Returns the current observation (temperature)."""
        return np.array([self.state[0]], dtype=np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """
        Render the environment's current state to a window.

        This method visualizes the current temperature state of the environment using a simple thermometer
        representation. The mercury level in the thermometer increases or decreases in accordance with the
        current normalized temperature value, providing a visual indication of temperature changes over time.

        Returns:
            np.ndarray or None
                - If self.render_mode is 'rgb_array', returns an RGB array of the screen.
                - If self.render_mode is 'human', returns None.
        """

        screen_width = 600
        screen_height = 400
        thermometer_height = 300
        thermometer_width = 50
        mercury_width = 30
        base_height = 10

        temp_range = self.max_temperature - self.min_temperature
        mercury_height = (
            (self.state[0] - self.min_temperature) / temp_range
        ) * thermometer_height

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    (screen_width, screen_height)
                )
            else:  # For rgb_array render mode, we don't need to display the window
                self.screen = pygame.Surface((screen_width, screen_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 16)  # Initialize the font

        self.screen.fill((255, 255, 255))  # Fill the background with white

        # Draw Thermometer
        thermometer_rect = pygame.Rect(
            (screen_width / 2) - (thermometer_width / 2),
            (screen_height / 2) - (thermometer_height / 2),
            thermometer_width,
            thermometer_height,
        )
        pygame.draw.rect(
            self.screen, (200, 200, 200), thermometer_rect
        )  # Light gray

        # Draw Mercury
        mercury_rect = pygame.Rect(
            (screen_width / 2) - (mercury_width / 2),
            (screen_height / 2) + (thermometer_height / 2) - mercury_height,
            mercury_width,
            mercury_height,
        )
        pygame.draw.rect(self.screen, (255, 0, 0), mercury_rect)  # Red

        # Draw Base
        base_rect = pygame.Rect(
            (screen_width / 2) - (thermometer_width / 2),
            (screen_height / 2) + (thermometer_height / 2),
            thermometer_width,
            base_height,
        )
        pygame.draw.rect(self.screen, (150, 150, 150), base_rect)  # Dark gray

        # Calculate the position for the observed mark line
        observed_ratio = (321.75 - 273.15) / (380 - 273.15)
        observed_mark_y = (screen_height / 2) + (thermometer_height / 2)
        observed_mark_y -= thermometer_height * observed_ratio

        observed_mark_start = (screen_width / 2) - (thermometer_width / 2)
        observed_mark_end = (screen_width / 2) + (thermometer_width / 2)

        # Draw the observed mark line
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (observed_mark_start, observed_mark_y),
            (observed_mark_end, observed_mark_y),
            5,
        )  # Black line

        # Draw temperature markings every x degrees from 273.15 K to 380 K
        min_temp_k = 273.15
        max_temp_k = 380
        temp_range_k = max_temp_k - min_temp_k
        marking_spacing_k = 20  # Every x degrees

        for temp_k in range(
            int(min_temp_k), int(max_temp_k) + 1, marking_spacing_k
        ):
            # Normalize the temperature to [0, 1] for the current scale
            normalized_temp = (temp_k - min_temp_k) / temp_range_k
            # Calculate the Y position for the marking based on the normalized temperature
            mark_y = (
                (screen_height / 2)
                + (thermometer_height / 2)
                - (normalized_temp * thermometer_height)
            )

            # Draw the marking line
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                ((screen_width / 2) - (thermometer_width / 2) - 10, mark_y),
                ((screen_width / 2) - (thermometer_width / 2), mark_y),
                2,
            )

            # Render the temperature text
            temp_text = self.font.render(f"{temp_k} K", True, (0, 0, 0))
            self.screen.blit(
                temp_text,
                (
                    (screen_width / 2) - (thermometer_width / 2) - 60,
                    mark_y - 10,
                ),
            )

        # Display Thermometer
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen).transpose([1, 0, 2])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
