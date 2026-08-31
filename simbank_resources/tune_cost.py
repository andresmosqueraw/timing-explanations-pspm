"""
Diagnose why PPO collapses to always-intervene on the SimBank Time-contact-HQ
log, and search for a `cost` value (in the Shoush & Dumas reward formula) that
makes always-wait, always-intervene, and the oracle genuinely separated
instead of intervene dominating almost everywhere.

Not part of the paper pipeline itself -- a one-off diagnostic/tuning script.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data" / "simbank_time_contact_hq_with_resources.pkl"


def reward(adapted: bool, ite: float, has_res: bool, cost: float, gain: float = 50.0, gain_res: float = 50.0) -> float:
    if adapted:
        if ite > 0:
            return (gain * ite) - cost + (gain_res if has_res else -gain_res)
        elif ite == 0:
            return -cost - gain_res
        else:
            return -cost - gain - gain_res
    else:
        if ite > 0:
            return -gain - gain_res
        elif ite == 0:
            return gain_res if has_res else 0.0
        else:
            return gain + gain_res


def main():
    df = pd.read_pickle(DATA)
    uq = df["unc_quality"].astype(float)
    ite_mean, ite_std = float(uq.mean()), float(uq.std()) or 1.0
    ite = (uq - ite_mean) / ite_std
    has_res = (df["available_resources"].astype(float) > 0).to_numpy()
    ite = ite.to_numpy()

    print(f"P(ite>0) = {(ite > 0).mean():.3f}, P(ite==0) = {(ite == 0).mean():.5f}, "
          f"P(ite<0) = {(ite < 0).mean():.3f}")
    print(f"P(has_res) = {has_res.mean():.3f}")
    print()

    print(f"{'cost':>6} | {'always-wait':>12} | {'always-interv':>14} | {'oracle':>10} | {'gap(oracle-wait)':>18}")
    for cost in [25, 40, 55, 70, 85, 100, 125, 150, 175, 200]:
        r_wait = np.array([reward(False, i, h, cost) for i, h in zip(ite, has_res)])
        r_int = np.array([reward(True, i, h, cost) for i, h in zip(ite, has_res)])
        r_oracle = np.maximum(r_wait, r_int)
        print(f"{cost:>6} | {r_wait.mean():>12.2f} | {r_int.mean():>14.2f} | {r_oracle.mean():>10.2f} | "
              f"{r_oracle.mean() - r_wait.mean():>18.2f}")


if __name__ == "__main__":
    main()
