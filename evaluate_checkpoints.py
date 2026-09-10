"""Score PPO checkpoints on the BPIC2017 gain sample (compute_gain_table's
protocol: 30,000 decision points, seed 42, resources at the training value),
for hyper-parameter sweeps of the coherent-state recipe.

Usage: python evaluate_checkpoints.py [--log BPIC2017] [--variant cate_retrained] models/variants/sweep/*.zip
"""
import argparse
import sys

import numpy as np
from stable_baselines3 import PPO

import paths
import pools

sys.path.insert(0, str(paths.REPO / "foreign"))
from train_ppo_fast_rl_prescriptive_monitoring import PPMEnvFast  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--log", default="BPIC2017")
    ap.add_argument("--variant", default=pools.DEFAULT_VARIANT)
    a = ap.parse_args()
    sample = 30000 if a.log == "BPIC2017" else None
    states, ite, hist = pools.bpic_full(paths.bpic_csv(a.log, a.variant), pools.treatment_col(a.log, a.variant), sample=sample, variant=a.variant)
    pool = [1] * pools.N_RESOURCES
    r_wait = np.mean([PPMEnvFast._reward(False, v, pool) for v in ite])
    rows = []
    for p in a.checkpoints:
        m = PPO.load(p, device="cpu")
        act, _ = m.predict(states, deterministic=True)
        g = np.mean([PPMEnvFast._reward(bool(x), v, pool) for x, v in zip(act, ite)])
        prec = float((ite[act == 1] > 0).mean()) if (act == 1).any() else float("nan")
        rows.append((p.split("/")[-1], float(act.mean()), prec, float((act[ite > 0] == 1).mean()), float(g)))
    print(f"{a.log}: always-wait {r_wait:.1f}")
    for name, rate, prec, rec, g in sorted(rows, key=lambda r: -r[4]):
        print(f"{name:48s} intervene {rate:.3f} precision {prec:.3f} recall {rec:.3f} gain {g:.1f}")


if __name__ == "__main__":
    main()
