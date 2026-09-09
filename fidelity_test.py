"""Deletion test on the timing justification (paper Section 7), on the same
evaluation pool the explanation cards and Table "card" (Section 6) are drawn
from: one prefix per case, 500 cases, seed 123, available_resources cycled
0..3 on the BPIC logs and read from the resource-augmented log on SimBank
(see pools.py for why).

Per log and per side: Integrated Gradients of the margin head against the
pool's mean state, then guided (top-|phi| features masked to the reference)
vs. random (20 draws) vs. anti-guided (bottom-|phi|) displacement of the
margin, for k = 1, 2. The wait side (WaitMarginHead, Delta Q_wait) is run on
the states where the policy waits; the intervene side (MarginHead, Delta Q)
on the states where it intervenes, whenever there are at least
``--min-states`` of them (the released checkpoints never intervene on the
BPIC pools; the "cate" variant does). The per-feature share of mean |phi| is
recorded alongside, so the feature the test "names" is on record.

Usage:
    python fidelity_test.py                       # the paper's policy (cate on BPIC, SimBank's checkpoint) -> fidelity_results.json
    python fidelity_test.py --variant released    # the released four-feature design -> fidelity_results_released.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from stable_baselines3 import PPO

import paths
import pools
from dual_level import MarginHead, WaitMarginHead, deletion_test, integrated_gradients


def _side(head, states: np.ndarray, reference: np.ndarray, feats: list[str], seed: int, label: str) -> dict:
    phi = integrated_gradients(head, states, reference, n_steps=128)
    mean_abs = np.abs(phi).mean(axis=0)
    share = mean_abs / mean_abs.sum()
    top1 = np.argmax(np.abs(phi), axis=1)
    with torch.no_grad():
        m = head(torch.from_numpy(states)).numpy()
    out = {
        "n_states": int(len(states)),
        "mean_abs_margin": float(np.abs(m).mean()),
        "phi_mean_abs": mean_abs.tolist(),
        "phi_share": share.tolist(),
        "phi_mean_signed": phi.mean(axis=0).tolist(),
        "top1_feature_counts": {f: int((top1 == i).sum()) for i, f in enumerate(feats)},
    }
    print(f"  [{label}] n={len(states)} E|margin|={out['mean_abs_margin']:.2f} share of mean|phi|:",
          {f: f"{s:.1%}" for f, s in zip(feats, share)})
    for k in (1, 2):
        res = deletion_test(head, states, phi, reference, k=k, n_random=20, seed=seed, track_sign_flips=True)
        out[f"k{k}"] = res.as_dict()
        print(f"    k={k}: guided={res.abs_guided:.4f} random={res.abs_random:.4f} anti={res.abs_anti:.4f} "
              f"gap={res.gap:.4f} (SE {res.gap_se:.4f}, z={res.gap / res.gap_se:.1f}) flip_guided={res.flip_guided:.3f}")
    return out


def run_one(name: str, model_path, states: np.ndarray, feats: list[str], seed: int = pools.POOL_SEED,
            max_states: int = pools.N_CASES, min_states: int = 30) -> dict:
    model = PPO.load(str(model_path), device="cpu")
    margin = MarginHead(model.policy, intervene_action=1)
    wait_head = WaitMarginHead(margin)
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        m0 = margin(torch.from_numpy(states)).numpy()
    wait_states, int_states = states[m0 <= 0], states[m0 > 0]
    if len(wait_states) > max_states:
        wait_states = wait_states[rng.choice(len(wait_states), size=max_states, replace=False)]
    if len(int_states) > max_states:
        int_states = int_states[rng.choice(len(int_states), size=max_states, replace=False)]
    reference = states.mean(axis=0)

    out = {
        "log": name, "model": str(model_path), "feature_names": feats,
        "n_pool_states": int(len(states)), "n_wait_states": int(len(wait_states)), "n_intervene_states": int(len(int_states)),
        "intervene_rate": float((m0 > 0).mean()), "reference": reference.tolist(),
    }
    print(f"\n{name}: pool={len(states)} wait={len(wait_states)} intervene={len(int_states)} intervene_rate={out['intervene_rate']:.4f}")
    print("  reference:", {f: round(v, 3) for f, v in zip(feats, reference.tolist())})
    if len(wait_states) >= min_states:
        w = _side(wait_head, wait_states, reference, feats, seed, "wait side, Delta Q_wait")
        out.update({"mean_abs_wait_margin": w["mean_abs_margin"], "phi_mean_abs": w["phi_mean_abs"], "phi_share": w["phi_share"],
                    "top1_feature_counts": w["top1_feature_counts"], "k1": w["k1"], "k2": w["k2"]})  # legacy flat keys
        out["wait_side"] = w
    if len(int_states) >= min_states:
        out["intervene_side"] = _side(margin, int_states, reference, feats, seed, "intervene side, Delta Q")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default=pools.DEFAULT_VARIANT, choices=list(pools.VARIANTS),
                    help="policy variant (all three logs); default = the paper's policy")
    ap.add_argument("--logs", nargs="+", default=list(pools.LOGS), choices=list(pools.LOGS))
    ap.add_argument("--min-states", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results = []
    for name in args.logs:
        states, _rows, feats = pools.evaluation_pool(name, args.variant)
        results.append(run_one(name, paths.variant_model(name, args.variant), states, feats, min_states=args.min_states))
    out_path = paths.Path(args.out) if args.out else (paths.FIDELITY_JSON if args.variant == pools.DEFAULT_VARIANT
                                                      else paths.REPO / f"fidelity_results_{args.variant}.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_path}")
