"""Historical-action vs. policy gain (paper Section 5.3, Table "gain"),
recomputed independently rather than cited from Shoush & Dumas: for every
decision point of each log, the reward Shoush & Dumas's own formula would
have paid under (a) the action recorded in the log and (b) the action the
retrained checkpoint chooses there, with the row's own counterfactual
ite = y1 - y0 (BPIC) or the documented uncertainty proxy (SimBank).

Pools (see pools.py): every decision point, available_resources held at the
training value (3) on the BPIC logs; BPIC2017 is a seed-42 sample of 30,000
of its >1M decision points. SimBank evaluates every event, the convention
the checkpoint was trained under, sampled to BPIC2012's n. The variant
restricted to the log's recorded contact decisions (``--simbank-rows
decision``) is degenerate under the uncertainty proxy (see pools.py) and is
written to the JSON alongside as a diagnostic, never as the main row.

TrafficFines is excluded: its "treatment" (absence of "Add penalty") is
definitionally entangled with the "deviant" label through the process's own
control flow, so the causal-effect estimator's treated-arm model has no
label variance to fit. A structural property of that log's treatment
definition, not a data bug.

Usage: python compute_gain_table.py
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from stable_baselines3 import PPO

import paths
import pools

sys.path.insert(0, str(paths.REPO / "foreign"))
sys.path.insert(0, str(paths.REPO / "simbank_resources"))
from train_ppo_fast_rl_prescriptive_monitoring import PPMEnvFast  # noqa: E402
from train_ppo_simbank import SimBankHQEnvFast  # noqa: E402

sys.path.insert(0, str(paths.REPO / "sepsis_resources"))
from train_ppo_sepsis import SepsisEnv  # noqa: E402


def _report(name: str, r_hist: np.ndarray, r_policy: np.ndarray, hist: np.ndarray, policy: np.ndarray) -> dict:
    out = {
        "n": int(len(r_hist)),
        "hist_gain": float(r_hist.mean()),
        "policy_gain": float(r_policy.mean()),
        "delta": float(r_policy.mean() - r_hist.mean()),
        "policy_intervene_rate": float(policy.mean()),
        "hist_intervene_rate": float(hist.mean()),
    }
    print(f"\n=== {name} ===  n={out['n']}")
    print(f"historical-action gain (mean): {out['hist_gain']:.3f}   intervene rate {out['hist_intervene_rate']:.1%}")
    print(f"policy gain (mean):            {out['policy_gain']:.3f}   intervene rate {out['policy_intervene_rate']:.1%}")
    print(f"delta (policy - historical):   {out['delta']:.3f}")
    return out


def bpic_gain(name: str, csv_path, model_path, treatment_col: str, sample: int | None = None, variant: str = "released") -> dict:
    states, ite, hist = pools.bpic_full(csv_path, treatment_col, sample=sample, variant=variant)
    model = PPO.load(str(model_path), device="cpu")
    policy, _ = model.predict(states, deterministic=True)
    policy = policy.astype(int)
    pool = [1] * pools.N_RESOURCES  # has_res=True: the training-time resource pool
    r_hist = np.array([PPMEnvFast._reward(bool(a), v, pool) for a, v in zip(hist, ite)])
    r_policy = np.array([PPMEnvFast._reward(bool(a), v, pool) for a, v in zip(policy, ite)])
    out = _report(name, r_hist, r_policy, hist, policy)
    # how the policy's interventions line up with the rows where intervening pays (ite > 0)
    pos = ite > 0
    out["oracle_gain"] = float(np.maximum(r_hist * 0 + np.array([PPMEnvFast._reward(True, v, pool) for v in ite]),
                                          np.array([PPMEnvFast._reward(False, v, pool) for v in ite])).mean())
    out["always_wait_gain"] = float(np.array([PPMEnvFast._reward(False, v, pool) for v in ite]).mean())
    out["always_intervene_gain"] = float(np.array([PPMEnvFast._reward(True, v, pool) for v in ite]).mean())
    out["share_ite_positive"] = float(pos.mean())
    out["policy_precision"] = float(pos[policy == 1].mean()) if (policy == 1).any() else None
    out["policy_recall"] = float((policy[pos] == 1).mean()) if pos.any() else None
    print(f"oracle gain {out['oracle_gain']:.3f}  always-wait {out['always_wait_gain']:.3f}  always-intervene {out['always_intervene_gain']:.3f}  "
          f"ite>0 share {out['share_ite_positive']:.3f}  policy precision {out['policy_precision']}  recall {out['policy_recall']}")
    return out


def sepsis_gain(variant: str = pools.DEFAULT_VARIANT) -> dict:
    """Every decision point of the Sepsis test split under the Sepsis reward (no
    resource term). Decision points precede the treatment, so the recorded
    action at each of them is to wait."""
    states, ite, _hist = pools.sepsis_full(variant=variant)
    model = PPO.load(str(paths.variant_model("Sepsis", variant)), device="cpu")
    policy = model.predict(states, deterministic=True)[0].astype(int)
    r_wait = np.array([SepsisEnv._reward(False, v) for v in ite])
    r_int = np.array([SepsisEnv._reward(True, v) for v in ite])
    r_policy = np.where(policy == 1, r_int, r_wait)
    out = _report("Sepsis", r_wait, r_policy, np.zeros_like(policy), policy)
    pos = ite > 0
    out.update({"always_wait_gain": float(r_wait.mean()), "always_intervene_gain": float(r_int.mean()), "oracle_gain": float(np.maximum(r_wait, r_int).mean()),
                "share_ite_positive": float(pos.mean()), "policy_precision": float(pos[policy == 1].mean()) if (policy == 1).any() else None,
                "policy_recall": float((policy[pos] == 1).mean()) if pos.any() else None})
    print(f"oracle gain {out['oracle_gain']:.3f}  always-wait {out['always_wait_gain']:.3f}  always-intervene {out['always_intervene_gain']:.3f}  "
          f"policy precision {out['policy_precision']}  recall {out['policy_recall']}")
    return out


def simbank_gain(rows: str, sample: int | None, variant: str = pools.DEFAULT_VARIANT) -> dict:
    states, ite, hist, has_res = pools.simbank_full(paths.simbank_pkl(variant), rows=rows, sample=sample, variant=variant)
    model = PPO.load(str(paths.variant_model("SimBank", variant)), device="cpu")
    policy, _ = model.predict(states, deterministic=True)
    policy = policy.astype(int)
    r_hist = np.array([SimBankHQEnvFast._reward(bool(a), v, bool(h)) for a, v, h in zip(hist, ite, has_res)])
    r_policy = np.array([SimBankHQEnvFast._reward(bool(a), v, bool(h)) for a, v, h in zip(policy, ite, has_res)])
    out = _report(f"SimBank Time-contact-HQ ({rows} rows)", r_hist, r_policy, hist, policy)
    r_wait = np.array([SimBankHQEnvFast._reward(False, v, bool(h)) for v, h in zip(ite, has_res)])
    r_int = np.array([SimBankHQEnvFast._reward(True, v, bool(h)) for v, h in zip(ite, has_res)])
    out["always_wait_gain"], out["always_intervene_gain"] = float(r_wait.mean()), float(r_int.mean())
    out["oracle_gain"] = float(np.maximum(r_wait, r_int).mean())
    pos = ite > 0
    out["share_ite_positive"] = float(pos.mean())
    out["policy_precision"] = float(pos[policy == 1].mean()) if (policy == 1).any() else None
    out["policy_recall"] = float((policy[pos] == 1).mean()) if pos.any() else None
    print(f"oracle gain {out['oracle_gain']:.3f}  always-wait {out['always_wait_gain']:.3f}  always-intervene {out['always_intervene_gain']:.3f}  "
          f"ite>0 share {out['share_ite_positive']:.3f}  policy precision {out['policy_precision']}  recall {out['policy_recall']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--simbank-rows", choices=["all", "decision"], default="all",
                    help="which SimBank rows count as decision points for the main table")
    ap.add_argument("--variant", default=pools.DEFAULT_VARIANT, choices=list(pools.VARIANTS),
                    help="policy variant (all three logs); default = the paper's policy")
    args = ap.parse_args()

    results = {"variant": args.variant}
    results["bpic2012"] = bpic_gain("BPIC2012", paths.bpic_csv("BPIC2012", args.variant), paths.variant_model("BPIC2012", args.variant),
                                    pools.treatment_col("BPIC2012", args.variant), variant=args.variant)
    results["bpic2017"] = bpic_gain("BPIC2017", paths.bpic_csv("BPIC2017", args.variant), paths.variant_model("BPIC2017", args.variant),
                                    pools.treatment_col("BPIC2017", args.variant), sample=30000, variant=args.variant)
    # Out of sample: the validation cases, which the agent never trained on.
    if args.variant == pools.DEFAULT_VARIANT:
        for lg, key, sample in (("BPIC2012", "bpic2012_val", None), ("BPIC2017", "bpic2017_val", 30000)):
            if paths.retrained_csv(lg, "val").exists():
                results[key] = bpic_gain(f"{lg} (validation cases, out of sample)", paths.retrained_csv(lg, "val"), paths.variant_model(lg, args.variant),
                                         pools.treatment_col(lg, args.variant), sample=sample, variant=args.variant)
                results[key]["cases"] = int(pools.load_bpic(paths.retrained_csv(lg, "val"))["case_id"].nunique())
    if paths.variant_model("Sepsis", args.variant).exists():
        results["sepsis"] = sepsis_gain(args.variant)
    n_match = results["bpic2012"]["n"]
    if paths.variant_model("SimBank", args.variant).exists():
        results["simbank"] = simbank_gain(args.simbank_rows, sample=n_match, variant=args.variant)
        results["simbank"]["rows"] = args.simbank_rows
        other = "all" if args.simbank_rows == "decision" else "decision"
        results[f"simbank_{other}_rows"] = simbank_gain(other, sample=n_match, variant=args.variant)
        results[f"simbank_{other}_rows"]["rows"] = other
    else:
        print(f"\nSimBank: no checkpoint for variant {args.variant}, skipped")
    out_path = paths.GAIN_JSON if args.variant == pools.DEFAULT_VARIANT else paths.REPO / f"gain_table_results_{args.variant}.json"

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_path}")
