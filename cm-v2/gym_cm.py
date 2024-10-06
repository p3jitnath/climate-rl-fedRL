import cm
import gymnasium as gym
import numpy as np
import redis


class ClimateModelEnv(gym.Env):
    """
    OpenAI Gym environment for interacting with a Fortran climate model.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        """
        Initialize the climate environment.
        """
        super(ClimateModelEnv, self).__init__()

        # Connect to Redis to store state and parameters
        self.r = redis.Redis(host="localhost", port=6379, db=0)

        # Define the action space (e.g., adjustments to forcing and feedback factor)
        # Actions are represented as continuous values: [delta_forcing, delta_feedback_factor]
        self.action_space = gym.spaces.Box(
            low=np.array([-0.5, -0.1]),
            high=np.array([0.5, 0.1]),
            dtype=np.float32,
        )

        # Define the observation space (e.g., temperature, forcing, feedback factor)
        # Observations include the current temperature, forcing, feedback factor, and heat capacity
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        # Initialize the climate system
        self.reset()

    def _get_info(self):
        return {"_": None}

    def step(self, action):
        # Parse the action (change in forcing, change in feedback factor)
        delta_forcing, delta_feedback_factor = action

        # Read current parameters from Redis
        forcing = float(self.r.get("forcing"))
        feedback_factor = float(self.r.get("feedback_factor"))
        heat_capacity = float(self.r.get("heat_capacity"))
        temperature_t = float(self.r.get("temperature_t"))

        # Apply the action (adjust forcing and feedback factor)
        forcing += delta_forcing
        feedback_factor += delta_feedback_factor

        # Run one step of the Fortran climate model
        temperature_t1 = cm.forward(
            forcing, feedback_factor, heat_capacity, temperature_t
        )

        # Store the updated temperature back in Redis
        self.r.set("temperature_t", temperature_t1)

        # Update Redis with the new forcing and feedback factor
        self.r.set("forcing", forcing)
        self.r.set("feedback_factor", feedback_factor)

        # Define a simple reward function (e.g., difference from room temperature)
        # This can be more complex based on the desired behavior of the system
        costs = (temperature_t1 - 25) ** 2

        # Return the observation, reward, done, and additional info
        obs = np.array(
            [temperature_t1, forcing, feedback_factor, heat_capacity]
        )

        return obs, -costs, False, False, self._get_info()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset parameters in Redis
        initial_forcing = 3.7
        initial_feedback_factor = 0.5
        initial_heat_capacity = 1000.0
        initial_temperature = 15.0

        self.r.set("forcing", initial_forcing)
        self.r.set("feedback_factor", initial_feedback_factor)
        self.r.set("heat_capacity", initial_heat_capacity)
        self.r.set("temperature_t", initial_temperature)

        # Return the initial observation
        obs = np.array(
            [
                initial_temperature,
                initial_forcing,
                initial_feedback_factor,
                initial_heat_capacity,
            ]
        )
        return obs, self._get_info()

    def render(self, mode="human"):
        # Read current state from Redis
        temperature_t = float(self.r.get("temperature_t"))
        forcing = float(self.r.get("forcing"))
        feedback_factor = float(self.r.get("feedback_factor"))
        heat_capacity = float(self.r.get("heat_capacity"))

        print(
            f"Temperature: {temperature_t} °C, Forcing: {forcing} W/m², Feedback Factor: {feedback_factor}, Heat Capacity: {heat_capacity}"
        )


# Create an instance of the climate environment
env = ClimateModelEnv()

# Run a simple loop interacting with the environment
obs = env.reset()
for _ in range(10):
    action = env.action_space.sample()  # Sample a random action
    obs, reward, _, _, info = env.step(action)
    env.render()  # Print the current state
