import glob
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import tyro


@dataclass
class Args:
    exp_id: str = "ebm-v3-fedRL-L-20k-a6-fed05"
    """name of the experiment"""


args = tyro.cli(Args)

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"
ENV_DIR = f"{BASE_DIR}/param_tune/results/{args.exp_id}"

best_metrics = defaultdict(dict)

for fn in sorted(glob.glob(f"{ENV_DIR}/best_*.pkl")):
    algo, cid, kind, date = fn.split("/")[-1][:-4].split("_")[1:]
    cid = int(cid)

    with open(fn, "rb") as file:
        metrics = pickle.load(file)

    subset_dict = {}
    subset_dict = metrics["config"]["params"]
    subset_dict["algo"] = algo
    subset_dict["cid"] = cid
    subset_dict["date"] = date
    subset_dict["episodic_return"] = float(metrics["last_episodic_return"])

    df = pd.DataFrame([subset_dict]).T
    df.columns = ["value"]

    print("-" * 32)
    print(df)
    print("-" * 32)

    best_metrics[algo][cid] = subset_dict

with open(f"{ENV_DIR}/best_results.json", "w") as file:
    json.dump(best_metrics, file, ensure_ascii=False, indent=4)
