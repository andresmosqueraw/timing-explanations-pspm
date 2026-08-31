"""
Padella-style Table II equivalent, recomputed independently (not cited from
Shoush & Dumas's own paper): for each of the 3 retained logs, compare the
mean gain (Shoush & Dumas's own reward formula, applied off-policy via the
row's counterfactual ite = y1 - y0, or the documented proxy for SimBank)
under the historically-recorded action versus under our trained PPO policy's
action, at every decision-point row.

TrafficFines is excluded: its "treatment" (absence of "Add penalty") is
definitionally entangled with the "deviant" label via the process's own
control-flow (see dataset-prep notebook), not a real independent lever, so
CausalLift's treated-arm model has zero label variance to fit. Not a data
or modeling bug we can fix -- a structural property of that log's own
treatment definition.

Usage: python compute_gain_table.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

GEN = Path(__file__).resolve().parent
FOREIGN = Path(
    "/home/andrew/Documents/docs/2-resolver-problema/process-mining/algorithms-explainability/"
    "faithful-pspm-explanations/experiments/generality_ppo/foreign"
)
sys.path.insert(0, str(FOREIGN))
from train_ppo_fast_rl_prescriptive_monitoring import PPMEnvFast, PPMEnvFast as _PPM  # noqa: E402

sys.path.insert(0, str(GEN / "simbank_resources"))
from train_ppo_simbank import SimBankHQEnvFast  # noqa: E402


def bpi_style_gain(name, csv_path, model_path, treatment_col, n_resources=3, sample=None):
    env = PPMEnvFast(csv_path, resources=n_resources)
    model = PPO.load(str(model_path), device="cpu")
    df = env._df

    ite = df["y1"].astype(float) - df["y0"].astype(float)
    hist_action = df[treatment_col].astype(int).to_numpy()

    states = np.stack(
        [
            (df["prefix_nr"].astype(float) / df["case_length"].astype(float).clip(lower=1)).clip(0, 1),
            df["reliability"].astype(float),
            df["deviation"].astype(float),
            np.full(len(df), float(n_resources)),
        ],
        axis=1,
    ).astype(np.float32)

    if sample is not None and len(states) > sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(states), size=sample, replace=False)
        states, ite, hist_action = states[idx], ite.to_numpy()[idx], hist_action[idx]
    else:
        ite = ite.to_numpy()

    policy_action, _ = model.predict(states, deterministic=True)
    policy_action = policy_action.astype(int)

    r_hist = np.array([PPMEnvFast._reward(bool(a), v, [1] * n_resources) for a, v in zip(hist_action, ite)])
    r_policy = np.array([PPMEnvFast._reward(bool(a), v, [1] * n_resources) for a, v in zip(policy_action, ite)])

    print(f"\n=== {name} ===")
    print(f"n = {len(ite)}")
    print(f"historical-action gain (mean): {r_hist.mean():.3f}")
    print(f"RL-policy gain (mean):         {r_policy.mean():.3f}")
    print(f"Delta (policy - historical):   {r_policy.mean() - r_hist.mean():.3f}")
    print(f"RL policy intervene rate:      {policy_action.mean():.1%}")
    print(f"historical intervene rate:     {hist_action.mean():.1%}")
    return {
        "n": len(ite),
        "hist_gain": float(r_hist.mean()),
        "policy_gain": float(r_policy.mean()),
        "delta": float(r_policy.mean() - r_hist.mean()),
        "policy_intervene_rate": float(policy_action.mean()),
        "hist_intervene_rate": float(hist_action.mean()),
    }


def simbank_gain(sample=28692):
    env = SimBankHQEnvFast(GEN / "simbank_resources" / "data" / "simbank_time_contact_hq_with_resources.pkl")
    model = PPO.load(str(GEN / "simbank_resources" / "models" / "ppo_simbank_time_contact_hq.zip"), device="cpu")
    df = env._df

    ite = df["ite_proxy"].astype(float).to_numpy()
    has_res = (df["available_resources"].astype(float) > 0).to_numpy()
    hist_action = (df["activity"] == "contact_headquarters").astype(int).to_numpy()

    states = np.stack(
        [
            (df["prefix_nr"].astype(float) / df["case_length"].astype(float).clip(lower=1)).clip(0, 1).to_numpy(),
            df["reliability"].astype(float).to_numpy(),
            df["deviation"].astype(float).to_numpy(),
            df["available_resources"].astype(float).to_numpy(),
        ],
        axis=1,
    ).astype(np.float32)

    if sample is not None and len(states) > sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(states), size=sample, replace=False)
        states, ite, hist_action, has_res = states[idx], ite[idx], hist_action[idx], has_res[idx]

    policy_action, _ = model.predict(states, deterministic=True)
    policy_action = policy_action.astype(int)

    r_hist = np.array([SimBankHQEnvFast._reward(bool(a), v, bool(h)) for a, v, h in zip(hist_action, ite, has_res)])
    r_policy = np.array([SimBankHQEnvFast._reward(bool(a), v, bool(h)) for a, v, h in zip(policy_action, ite, has_res)])

    print("\n=== SimBank (Time contact HQ) ===")
    print(f"n = {len(ite)}")
    print(f"historical-action gain (mean): {r_hist.mean():.3f}")
    print(f"RL-policy gain (mean):         {r_policy.mean():.3f}")
    print(f"Delta (policy - historical):   {r_policy.mean() - r_hist.mean():.3f}")
    print(f"RL policy intervene rate:      {policy_action.mean():.1%}")
    print(f"historical intervene rate:     {hist_action.mean():.1%}")
    return {
        "n": len(ite),
        "hist_gain": float(r_hist.mean()),
        "policy_gain": float(r_policy.mean()),
        "delta": float(r_policy.mean() - r_hist.mean()),
        "policy_intervene_rate": float(policy_action.mean()),
        "hist_intervene_rate": float(hist_action.mean()),
    }


if __name__ == "__main__":
    results = {}
    results["bpic2012"] = bpi_style_gain(
        "BPIC2012",
        GEN / "data" / "ready_to_use_adaptive_bpic2012.csv",
        GEN / "models" / "ppo_bpic2012_rl_prescriptive_monitoring.zip",
        "Treatment",
        n_resources=3,
    )
    results["bpic2017"] = bpi_style_gain(
        "BPIC2017",
        Path(
            "/home/andrew/Documents/docs/2-resolver-problema/process-mining/algorithms-explainability/"
            "libraries-prescriptive-process/01-rl-online/RL-prescriptive-monitoring/rl/data/ready_to_use_adaptive_bpic2017.csv"
        ),
        GEN / "models" / "ppo_bpic2017_rl_prescriptive_monitoring.zip",
        "treatment",
        n_resources=3,
        sample=30000,
    )
    results["simbank"] = simbank_gain()

    import json
    out_path = GEN / "gain_table_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_path}")
