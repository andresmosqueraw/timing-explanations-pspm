"""
Fidelity evaluation of the wait-vs-act timing justification phi^{Delta Q_wait}
(paper1's Section 4), reusing the deletion-test protocol already built and
validated in faithful-pspm-explanations/dual_level.py (paper2's repo): guided
(top-|phi| features masked to the reference) vs. random vs. anti-guided
(bottom-|phi| features masked) displacement of WaitMarginHead(s), on the
wait-state pool of each of the 3 checkpoints used in paper1.

This does NOT run the other attribution methods or the risk-side (CriticHead)
test from dual_level.py -- paper1's scope is only phi^{Delta Q_wait} via
Integrated Gradients (see paper1/4_FrameworkForExplainingTiming.tex).

Usage: python fidelity_test.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

GEN = Path(__file__).resolve().parent
sys.path.insert(0, str(GEN))
from dual_level import MarginHead, WaitMarginHead, integrated_gradients, deletion_test  # noqa: E402

FOREIGN = Path(
    "/home/andrew/Documents/docs/2-resolver-problema/process-mining/algorithms-explainability/"
    "faithful-pspm-explanations/experiments/generality_ppo/foreign"
)
sys.path.insert(0, str(FOREIGN))
from train_ppo_fast_rl_prescriptive_monitoring import PPMEnvFast  # noqa: E402

sys.path.insert(0, str(GEN / "simbank_resources"))
from train_ppo_simbank import SimBankHQEnvFast  # noqa: E402

FEATS = ["relative_position", "reliability", "deviation", "available_resources"]


def bpi_states(csv_path, n_resources=3, sample=None, seed=42):
    env = PPMEnvFast(csv_path, resources=n_resources)
    df = env._df
    rel = (df["prefix_nr"].astype(float) / df["case_length"].astype(float).clip(lower=1)).clip(0, 1)
    states = np.stack(
        [rel.to_numpy(), df["reliability"].astype(float).to_numpy(),
         df["deviation"].astype(float).to_numpy(),
         np.full(len(df), float(n_resources))],
        axis=1,
    ).astype(np.float32)
    if sample is not None and len(states) > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(states), size=sample, replace=False)
        states = states[idx]
    return states


def simbank_states(seed=42):
    env = SimBankHQEnvFast(GEN / "simbank_resources" / "data" / "simbank_time_contact_hq_with_resources.pkl")
    df = env._df
    rel = (df["prefix_nr"].astype(float) / df["case_length"].astype(float).clip(lower=1)).clip(0, 1)
    states = np.stack(
        [rel.to_numpy(), df["reliability"].astype(float).to_numpy(),
         df["deviation"].astype(float).to_numpy(),
         df["available_resources"].astype(float).to_numpy()],
        axis=1,
    ).astype(np.float32)
    return states


def run_one(name, model_path, states, seed=42, max_wait_states=500):
    model = PPO.load(str(model_path), device="cpu")
    policy = model.policy
    margin = MarginHead(policy, intervene_action=1)
    wait_head = WaitMarginHead(margin)

    import torch
    dev = next(policy.parameters()).device
    with torch.no_grad():
        m0 = margin(torch.from_numpy(states).to(dev)).cpu().numpy()
    wait_states = states[m0 < 0]
    n_wait = len(wait_states)
    if n_wait > max_wait_states:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_wait, size=max_wait_states, replace=False)
        wait_states = wait_states[idx]

    reference = states.mean(axis=0)
    phi = integrated_gradients(wait_head, wait_states, reference, n_steps=128)

    out = {"log": name, "n_wait_states": int(len(wait_states)), "n_total_states": int(len(states)),
           "reference": reference.tolist(), "feature_names": FEATS}
    for k in (1, 2):
        res = deletion_test(wait_head, wait_states, phi, reference, k=k, n_random=20, seed=seed,
                             track_sign_flips=True)
        out[f"k{k}"] = res.as_dict()
        print(f"{name} k={k}: guided={res.abs_guided:.4f} random={res.abs_random:.4f} "
              f"anti={res.abs_anti:.4f} gap={res.gap:.4f} (SE {res.gap_se:.4f}) "
              f"flip_guided={res.flip_guided:.1%} flip_random={res.flip_random:.1%}")
    return out


if __name__ == "__main__":
    results = []

    bpic2012_states = bpi_states(GEN / "data" / "ready_to_use_adaptive_bpic2012.csv", n_resources=3)
    results.append(run_one(
        "BPIC2012",
        GEN / "models" / "ppo_bpic2012_rl_prescriptive_monitoring.zip",
        bpic2012_states,
    ))

    bpic2017_csv = Path(
        "/home/andrew/Documents/docs/2-resolver-problema/process-mining/algorithms-explainability/"
        "libraries-prescriptive-process/01-rl-online/RL-prescriptive-monitoring/rl/data/ready_to_use_adaptive_bpic2017.csv"
    )
    bpic2017_states = bpi_states(bpic2017_csv, n_resources=3, sample=30000)
    results.append(run_one(
        "BPIC2017",
        GEN / "models" / "ppo_bpic2017_rl_prescriptive_monitoring.zip",
        bpic2017_states,
    ))

    sb_states = simbank_states()
    results.append(run_one(
        "SimBank",
        GEN / "simbank_resources" / "models" / "ppo_simbank_time_contact_hq.zip",
        sb_states,
    ))

    out_path = GEN / "fidelity_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_path}")
