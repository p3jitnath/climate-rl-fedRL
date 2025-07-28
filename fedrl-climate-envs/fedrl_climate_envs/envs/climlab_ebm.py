import os
import time
from dataclasses import dataclass

import climlab
import numpy as np
import tyro
import xarray as xr
from smartredis import Client

WAIT_TIME = 1e-4
EBM_LATITUDES = 96


@dataclass
class Args:
    num_clients: int = 2
    """number of clients (corresponding to latitudes in the EBM)"""


class Utils:

    BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"
    DATASETS_DIR = f"{BASE_DIR}/datasets"

    fp_Ts = f"{DATASETS_DIR}/skt.sfc.mon.1981-2010.ltm.nc"
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
    ).sortby("lat")

    lat_ncep = ncep_Ts.lat
    lon_ncep = ncep_Ts.lon
    Ts_ncep_annual = ncep_Ts.skt.mean(dim=("lon", "time"))

    a0_ref = 0.354
    a2_ref = 0.25
    D_ref = 0.6
    A_ref = 2.1
    B_ref = 2

    REDIS_ADDRESS = os.getenv("SSDB")
    if REDIS_ADDRESS is None:
        raise EnvironmentError("SSDB environment variable is not set.")
    redis = Client(address=REDIS_ADDRESS, cluster=False)
    print(f"[RL Env] Connected to Redis server: {REDIS_ADDRESS}")


class ClimLabEBM:

    def __init__(self, utils):
        self.utils = utils
        self.ebm = climlab.EBM_annual(
            a0=self.utils.a0_ref,
            a2=self.utils.a2_ref,
            D=self.utils.D_ref,
            A=np.array(
                [self.utils.A_ref * 1e2 for x in range(EBM_LATITUDES)]
            ).reshape(-1, 1),
            B=np.array(
                [self.utils.B_ref for x in range(EBM_LATITUDES)]
            ).reshape(-1, 1),
            num_lat=EBM_LATITUDES,
            name="EBM Model w/ RL",
        )
        self.ebm.Ts[:] = 50.0
        self.Ts_ncep_annual = np.array(
            self.utils.Ts_ncep_annual.interp(
                lat=self.ebm.lat, kwargs={"fill_value": "extrapolate"}
            )
        ).reshape(-1, 1)

        self.climlab_ebm = climlab.process_like(self.ebm)
        self.climlab_ebm.name = "EBM Model"


class Exists:

    def __init__(self):
        self.sigcompute = np.zeros(args.num_clients)
        self.sigstart = np.zeros(args.num_clients)
        self.params = np.zeros(args.num_clients)


args = tyro.cli(Args)
utils, exists = Utils(), Exists()
ctr = 0

while True:

    # 0. Check which key exists
    for idx, cid in enumerate(range(args.num_clients)):
        if utils.redis.tensor_exists(f"SIGSTART_S{cid}"):
            exists.sigstart[idx] = 1
        if utils.redis.tensor_exists(f"SIGCOMPUTE_S{cid}"):
            exists.sigcompute[idx] = 1

    # if ctr % int(1e6) == 0:
    #     print(f"[climlab EBM] exists.sigstart: {exists.sigstart}", flush=True)
    #     print(f"[climlab EBM] exists.sigcompute: {exists.sigcompute}", flush=True)

    if np.sum(exists.sigstart) > 0:
        cebm = ClimLabEBM(utils)
        cebm.EBM_SUBLATITUDES = EBM_LATITUDES // args.num_clients

        # 1. Send the initialised state and other reference variables to the RL agent
        for idx, cid in enumerate(range(args.num_clients)):
            if exists.sigstart[idx] != 0:
                # print(f"[climlab EBM] deleted: SIGSTART_S{cid}", flush=True)
                utils.redis.delete_tensor(f"SIGSTART_S{cid}")
                time.sleep(WAIT_TIME)
                # print(f"[climlab EBM] sent: f2py_redis_s{cid}", flush=True)
                utils.redis.put_tensor(
                    f"f2py_redis_s{cid}",
                    np.array(
                        [
                            cebm.ebm.Ts,
                            cebm.climlab_ebm.Ts,
                            cebm.Ts_ncep_annual,
                            np.array(cebm.ebm.lat).reshape(-1, 1),
                        ],
                        dtype=np.float32,
                    ),
                )
                exists.sigstart[idx] = 0

    if np.sum(exists.sigcompute) == args.num_clients:
        params = [None for x in range(args.num_clients)]

        # 2. Wait for the RL agents to compute the new parameters and send
        while sum(exists.params) != args.num_clients:
            for idx, cid in enumerate(range(args.num_clients)):
                if params[idx] is None:
                    if utils.redis.tensor_exists(f"py2f_redis_s{cid}"):
                        # print(f"[climlab EBM] received: py2f_redis_s{cid}", flush=True)
                        params[idx] = utils.redis.get_tensor(
                            f"py2f_redis_s{cid}"
                        )
                        exists.params[idx] = 1
                        # print(f"[climlab EBM] exists.params: {exists.params}", flush=True)
                        time.sleep(WAIT_TIME)
                        # print(f"[climlab EBM] deleted: py2f_redis_s{cid}", flush=True)
                        utils.redis.delete_tensor(f"py2f_redis_s{cid}")
                    else:
                        continue  # Extract the params in another pass

        # 3. Perform model step
        params = np.array(params)
        D = np.mean(params[:, 0])
        A = np.array(params[:, 1 : cebm.EBM_SUBLATITUDES + 1]).reshape(-1, 1)
        B = np.array(params[:, cebm.EBM_SUBLATITUDES + 1 : -2]).reshape(-1, 1)
        a0 = np.mean(params[:, -2])
        a2 = np.mean(params[:, -1])

        cebm.ebm.subprocess["diffusion"].D = D
        cebm.ebm.subprocess["LW"].A = A
        cebm.ebm.subprocess["LW"].B = B
        cebm.ebm.subprocess["albedo"].a0 = a0
        cebm.ebm.subprocess["albedo"].a2 = a2

        cebm.ebm.step_forward()
        cebm.climlab_ebm.step_forward()

        # 4. Send the state to the RL agent
        for idx, cid in enumerate(range(args.num_clients)):
            # print(f"[climlab EBM] deleted: SIGCOMPUTE_S{cid}", flush=True)
            utils.redis.delete_tensor(f"SIGCOMPUTE_S{cid}")
            time.sleep(WAIT_TIME)
            # print(f"[climlab EBM] sent: f2py_redis_s{cid}", flush=True)
            utils.redis.put_tensor(
                f"f2py_redis_s{cid}",
                np.array([cebm.ebm.Ts, cebm.climlab_ebm.Ts], dtype=np.float32),
            )

        exists.params[:] = 0
        exists.sigcompute[:] = 0

    ctr += 1
    time.sleep(WAIT_TIME)  # wait for a bit before checking for signals again
