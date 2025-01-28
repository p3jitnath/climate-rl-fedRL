import os

import climlab
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from gymnasium import spaces
from matplotlib.gridspec import GridSpec


class Utils:

    BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl-f2py/cm-v6"
    DATASETS_DIR = f"{BASE_DIR}/datasets"

    fp_Ts = f"{DATASETS_DIR}/skt.sfc.mon.1981-2010.ltm.nc"
    fp_ulwrf = f"{DATASETS_DIR}/ulwrf.ntat.mon.1981-2010.ltm.nc"
    fp_dswrf = f"{DATASETS_DIR}/dswrf.ntat.mon.1981-2010.ltm.nc"
    fp_uswrf = f"{DATASETS_DIR}/uswrf.ntat.mon.1981-2010.ltm.nc"

    ncep_url = "http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/"

    def download_and_save_dataset(url, filepath, dataset_name):
        if not os.path.exists(filepath):
            print(f"Downloading {dataset_name} data ...")
            dataset = xr.open_dataset(url, decode_times=False)
            dataset.to_netcdf(filepath, format="NETCDF3_64BIT")
            print(f"{dataset_name} data saved to {filepath}")
        else:
            print(f"Loading {dataset_name} data ...")
            dataset = xr.open_dataset(
                filepath,
                decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
            )
        return dataset

    ncep_Ts = download_and_save_dataset(
        ncep_url + "surface_gauss/skt.sfc.mon.1981-2010.ltm.nc",
        fp_Ts,
        "NCEP surface temperature",
    )
    ncep_ulwrf = download_and_save_dataset(
        ncep_url + "other_gauss/ulwrf.ntat.mon.1981-2010.ltm.nc",
        fp_ulwrf,
        "NCEP upwelling longwave radiation",
    )
    ncep_dswrf = download_and_save_dataset(
        ncep_url + "other_gauss/dswrf.ntat.mon.1981-2010.ltm.nc",
        fp_dswrf,
        "NCEP downwelling shortwave radiation",
    )
    ncep_uswrf = download_and_save_dataset(
        ncep_url + "other_gauss/uswrf.ntat.mon.1981-2010.ltm.nc",
        fp_uswrf,
        "NCEP upwelling shortwave radiation",
    )

    lat_ncep = ncep_Ts.lat
    lon_ncep = ncep_Ts.lon
    Ts_ncep_annual = ncep_Ts.skt.mean(dim=("lon", "time"))

    OLR_ncep_annual = ncep_ulwrf.ulwrf.mean(dim=("lon", "time"))
    ASR_ncep_annual = (ncep_dswrf.dswrf - ncep_uswrf.uswrf).mean(
        dim=("lon", "time")
    )

    A = 210
    B = 2
    a0 = 0.354
    a2 = 0.25
    nMonthDays = 30


class EnergyBasedModelEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode=None, locale="uk"):
        self.min_D = 0  # no heat transport
        self.max_D = 2  # perfect heat transport

        self.min_temperature = -90
        self.max_temperature = 90

        self.utils = Utils()

        self.action_space = spaces.Box(
            low=np.array(
                [
                    self.min_D,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.max_D,
                ],
                dtype=np.float32,
            ),
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=self.min_temperature,
            high=self.max_temperature,
            shape=(len(self.utils.lat_ncep),),
            dtype=np.float32,
        )

        assert (
            render_mode is None or render_mode in self.metadata["render_modes"]
        )

        self.render_mode = render_mode
        self.reset()

    def _get_obs(self):
        return self._get_temp()

    def _get_temp(self, model="RL"):
        if model == "RL":
            ebm = self.ebm
        elif model == "climlab":
            ebm = self.climlab_ebm
        temp = np.array(ebm.Ts, dtype=np.float32).reshape(-1)
        return temp

    def _get_info(self):
        return {"_": None}

    def _get_params(self):
        D = self.ebm.subprocess["diffusion"].D
        params = np.array([D], dtype=np.float32)
        return params

    def _get_state(self):
        state = self._get_temp()
        return state

    def step(self, action):
        D = action[0]
        D = np.clip(D, self.min_D, self.max_D)

        self.ebm.subprocess["diffusion"].D = D

        self.ebm.integrate_days(self.utils.nMonthDays, verbose=False)
        self.climlab_ebm.integrate_days(self.utils.nMonthDays, verbose=False)

        Tprofile = self._get_temp()
        costs = np.mean(
            (Tprofile - self.utils.Ts_ncep_annual.values[::-1]) ** 2
        )

        self.state = self._get_state()

        return self._get_obs(), -costs, False, False, self._get_info()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ebm = climlab.EBM_annual(
            A=self.utils.A,
            B=self.utils.B,
            a0=self.utils.a0,
            a2=self.utils.a2,
            num_lat=len(self.utils.lat_ncep),
            name="EBM Model w/ RL",
        )

        # Initialize a climlab EBM model clone
        self.climlab_ebm = climlab.process_like(self.ebm)
        self.climlab_ebm.name = "EBM Model"

        self.state = self._get_state()
        return self._get_obs(), self._get_info()

    def _render_frame(self, save_fig=None, idx=None):
        fig = plt.figure(figsize=(28, 8))
        gs = GridSpec(1, 3, figure=fig)

        params = self._get_params()

        # Left subplot: diffusivity as bar plot
        ax1 = fig.add_subplot(gs[0, 0])

        ax1_labels = [
            "D",
        ]
        ax1_colors = [
            "tab:blue",
        ]
        ax1_bars = ax1.bar(
            ax1_labels,
            [
                params[0],
            ],
            color=ax1_colors,
            width=0.75,
        )
        ax1.set_ylim(0, 10)
        ax1.set_ylabel("Value", fontsize=14)
        ax1.set_title("Parameters")

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
        ax2.plot(self.ebm.lat, self.ebm.Ts, label="RCE Model w/ RL")
        ax2.plot(self.climlab_ebm.lat, self.climlab_ebm.Ts, label="RCE Model")
        ax2.plot(
            self.utils.lat_ncep,
            self.utils.Ts_ncep_annual,
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
            x=self.ebm.lat,
            height=np.abs(
                self.ebm.Ts.reshape(-1)
                - self.utils.Ts_ncep_annual.values[::-1]
            ),
            label="RCE Model w/ RL",
        )
        ax3.bar(
            x=self.climlab_ebm.lat,
            height=np.abs(
                self.climlab_ebm.Ts.reshape(-1)
                - self.utils.Ts_ncep_annual.values[::-1]
            ),
            label="RCE Model",
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
