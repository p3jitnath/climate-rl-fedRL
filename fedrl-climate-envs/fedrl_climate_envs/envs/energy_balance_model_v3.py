import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.gridspec import GridSpec
from smartredis import Client

EBM_LATITUDES = 96
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", 2))
EBM_SUBLATITUDES = EBM_LATITUDES // NUM_CLIENTS


class EnergyBalanceModelEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, seed=None, cid=None, render_mode=None):

        # self.max_D = self.min_D = (
        #     0.6  # Have to be kept constant for single latitude cases
        # )

        self.seed = seed
        self.cid = cid

        print(f"[RL Env] Environment ID: {self.cid}", flush=True)
        print(f"[RL Env] Number of clients: {NUM_CLIENTS}", flush=True)

        self.D = 0.6
        self.min_D = 0.55
        self.max_D = 0.65

        self.A = np.array([2.1 for x in range(EBM_SUBLATITUDES)])
        self.min_A = 1.4
        self.max_A = 4.2

        self.B = np.array([2 for x in range(EBM_SUBLATITUDES)])
        self.min_B = 1.95
        self.max_B = 2.05

        self.a0 = 0.354
        self.min_a0 = 0.3
        self.max_a0 = 0.4

        self.a2 = 0.25
        self.min_a2 = 0.2
        self.max_a2 = 0.3

        self.min_temperature = -90
        self.max_temperature = 90

        self.action_space = spaces.Box(
            low=np.array(
                [
                    self.min_D,
                    *[self.min_A for _ in range(EBM_SUBLATITUDES)],
                    *[self.min_B for _ in range(EBM_SUBLATITUDES)],
                    self.min_a0,
                    self.min_a2,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.max_D,
                    *[self.max_A for _ in range(EBM_SUBLATITUDES)],
                    *[self.max_B for _ in range(EBM_SUBLATITUDES)],
                    self.max_a0,
                    self.max_a2,
                ],
                dtype=np.float32,
            ),
            shape=(2 * EBM_SUBLATITUDES + 3,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=self.min_temperature,
            high=self.max_temperature,
            shape=(EBM_SUBLATITUDES,),
            dtype=np.float32,
        )

        assert (
            render_mode is None or render_mode in self.metadata["render_modes"]
        )

        self.render_mode = render_mode
        self.wait_time = 1e-4

        # SmartRedis setup
        self.REDIS_ADDRESS = os.getenv("SSDB")
        if self.REDIS_ADDRESS is None:
            raise EnvironmentError("SSDB environment variable is not set.")
        self.redis = Client(address=self.REDIS_ADDRESS, cluster=False)
        print(
            f"[RL Env] Connected to Redis server: {self.REDIS_ADDRESS}",
            flush=True,
        )

        # print(f"[RL Env] sent: SIGALIVE_S{self.cid}", flush=True)
        self.redis.put_tensor(
            f"SIGALIVE_S{self.cid}", np.array([1], dtype=np.int32)
        )

        # self.reset(self.seed)

    def _get_params(self):
        return np.array(
            [self.D, *(self.A), *(self.B), self.a0, self.a2], dtype=np.float32
        )

    def _get_obs(self):
        return np.array(self.state, dtype=np.float32)

    def _get_info(self):
        return {"_": None}

    def step(self, action):
        self.D = action[0]
        self.A, self.B = (
            action[1 : EBM_SUBLATITUDES + 1],
            action[EBM_SUBLATITUDES + 1 : -2],
        )
        self.a0, self.a2 = action[-2], action[-1]

        # Clip actions to the allowed range
        self.D = np.clip(self.D, self.min_D, self.max_D)
        self.A = np.clip(self.A, self.min_A, self.max_A)
        self.B = np.clip(self.B, self.min_B, self.max_B)
        self.a0 = np.clip(self.a0, self.min_a0, self.max_a0)
        self.a2 = np.clip(self.a2, self.min_a2, self.max_a2)

        # Send the parameters to Redis
        # print(f"[RL Env] sent: PY2F_REDIS_S{self.cid}", flush=True)
        self.redis.put_tensor(
            f"PY2F_REDIS_S{self.cid}",
            np.array(
                [self.D, *(self.A * 1e2), *(self.B), self.a0, self.a2],
                dtype=np.float32,
            ),
        )
        # print(f"[RL Env] sent: SIGCOMPUTE_S{self.cid}", flush=True)
        self.redis.put_tensor(
            f"SIGCOMPUTE_S{self.cid}", np.array([1], dtype=np.int32)
        )

        # Wait for the climlab model to compute the new ebm temperatures and send
        self.ebm_Ts = None
        while self.ebm_Ts is None:
            if self.redis.tensor_exists(f"F2PY_REDIS_S{self.cid}"):
                # print(f"[RL Env] received: F2PY_REDIS_S{self.cid}", flush=True)
                self.ebm_Ts, self.climlab_ebm_Ts = self.redis.get_tensor(
                    f"F2PY_REDIS_S{self.cid}"
                )
                time.sleep(self.wait_time)
                # print(f"[RL Env] deleted: F2PY_REDIS_S{self.cid}", flush=True)
                self.redis.delete_tensor(f"F2PY_REDIS_S{self.cid}")
            else:
                continue

        # Update the state
        self.state = self.ebm_Ts[self.ebm_min_idx : self.ebm_max_idx].reshape(
            -1
        )

        # Calculate the cost (mean squared error) in Python
        costs = np.mean(
            (self.ebm_Ts - self.Ts_ncep_annual)[
                self.ebm_min_idx : self.ebm_max_idx
            ]
            ** 2
        )

        return self._get_obs(), -costs, False, False, self._get_info()

    def get_target_state(self):
        return np.array(
            self.Ts_ncep_annual.values[self.ebm_min_idx : self.ebm_max_idx]
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Send the start signal to the Fortran model
        # print(f"[RL Env] sent: SIGSTART_S{self.cid}", flush=True)
        self.redis.put_tensor(
            f"SIGSTART_S{self.cid}", np.array([1], dtype=np.int32)
        )

        # Wait for the climlab model to compute the new temperature and send
        self.ebm_Ts = None
        while self.ebm_Ts is None:
            if self.redis.tensor_exists(f"F2PY_REDIS_S{self.cid}"):
                # print(f"[RL Env] received: F2PY_REDIS_S{self.cid}", flush=True)
                (
                    self.ebm_Ts,
                    self.climlab_ebm_Ts,
                    self.Ts_ncep_annual,
                    self.ebm_lat,
                ) = self.redis.get_tensor(f"F2PY_REDIS_S{self.cid}")
                time.sleep(self.wait_time)
                # print(f"[RL Env] deleted: F2PY_REDIS_S{self.cid}", flush=True)
                self.redis.delete_tensor(f"F2PY_REDIS_S{self.cid}")
            else:
                continue  # Wait for the computation to complete

        # Initialise the state
        self.ebm_min_idx, self.ebm_max_idx = (
            self.cid * EBM_SUBLATITUDES,
            (self.cid + 1) * EBM_SUBLATITUDES,
        )
        self.phi = self.ebm_lat[self.ebm_min_idx : self.ebm_max_idx]
        self.state = self.ebm_Ts[self.ebm_min_idx : self.ebm_max_idx].reshape(
            -1
        )

        return self._get_obs(), self._get_info()

    def _render_frame(self, save_fig=None, idx=None):
        fig = plt.figure(figsize=(28, 8))
        gs = GridSpec(1, 3, figure=fig)

        params = self._get_params()

        # Left subplot: diffusivity as bar plot
        ax1 = fig.add_subplot(gs[0, 0])

        ax1_labels = ["D", "A", "B", "a0", "a2"]
        ax1_colors = [
            "tab:blue",
            "tab:blue",
            "tab:blue",
            "tab:blue",
            "tab:blue",
        ]
        ax1_bars = ax1.bar(
            ax1_labels,
            [
                params[0],
                np.mean(params[1 : EBM_SUBLATITUDES + 1]),
                np.mean(params[EBM_SUBLATITUDES + 1 : -2]),
                *params[-2:],
            ],
            color=ax1_colors,
            width=0.75,
        )
        ax1.set_ylim(0, 10)
        ax1.set_ylabel("Value", fontsize=14)

        # Add values on top of the bars
        for bar in ax1_bars:
            height = bar.get_height()
            ax1.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        # Middle subplot: Temperature v/s Latitude
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(
            self.ebm_lat[self.ebm_min_idx : self.ebm_max_idx],
            self.ebm_Ts[self.ebm_min_idx : self.ebm_max_idx],
            label="EBM Model w/ RL",
        )
        ax2.plot(
            self.ebm_lat[self.ebm_min_idx : self.ebm_max_idx],
            self.climlab_ebm_Ts[self.ebm_min_idx : self.ebm_max_idx],
            label="EBM Model",
        )
        ax2.plot(
            self.ebm_lat,
            self.Ts_ncep_annual,
            label="Observations",
            c="k",
        )
        ax2.set_ylabel("Temperature (°C)")
        ax2.set_xlabel("Latitude")
        ax2.set_xlim(-90, 90)
        ax2.set_xticks(np.arange(-90, 91, 30))
        ax2.legend()
        ax2.grid()

        # Right subplot: Error v/s Latitude
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(
            x=self.ebm_lat[self.ebm_min_idx : self.ebm_max_idx].reshape(-1),
            height=np.abs(
                self.ebm_Ts[self.ebm_min_idx : self.ebm_max_idx]
                - self.Ts_ncep_annual[self.ebm_min_idx : self.ebm_max_idx]
            ).reshape(-1),
            label="EBM Model w/ RL",
        )
        ax3.bar(
            x=self.ebm_lat[self.ebm_min_idx : self.ebm_max_idx].reshape(-1),
            height=np.abs(
                self.climlab_ebm_Ts[self.ebm_min_idx : self.ebm_max_idx]
                - self.Ts_ncep_annual[self.ebm_min_idx : self.ebm_max_idx]
            ).reshape(-1),
            label="EBM Model",
            zorder=-1,
        )
        ax3.set_ylabel("Error  (°C)")
        ax3.set_xlabel("Latitude")
        ax3.set_xlim(-90, 90)
        ax3.set_xticks(np.arange(-90, 91, 30))
        ax3.legend()
        ax3.grid()

        return fig

    def render(self, **kwargs):
        if self.render_mode == "human":
            self._render_frame(**kwargs)
            plt.show()
        elif self.render_mode == "rgb_array":
            fig = self._render_frame(**kwargs)
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape((height, width, 3))
            plt.close(fig)
            return image
