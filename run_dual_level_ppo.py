"""Generality study (paper Sec. 4.5): dual-level (PL-xPsPM) explanations on
external SB3 PPO PsPM agents.

Targets:
- rl-prescriptive-monitoring (primary): 4-D state
  [relative_position, reliability, deviation, available_resources], BPIC 2017.
- when-to-treat (secondary): 3-D state [relative_position, lower_TE, upper_TE],
  BPIC 2017 counterfactual treatment-effect bounds.

The PPO checkpoints live in ./models (committed; retrained with
foreign/train_ppo_fast_*.py, 300k / 200k steps, seed 42). The state CSVs are
the third-party repos' own preprocessed data (478 MB / 186 MB — not committed);
point --lib (or XPPM_PPO_LIB) at a checkout that contains
  RL-prescriptive-monitoring/rl/data/ready_to_use_adaptive_bpic2017.csv
  WhenToTreat/RL/data/results_adaptive_counterfacs_bpic2017.csv
Both files are produced by the respective repos' preprocessing from the public
BPIC 2017 log; see README.md.

States are rebuilt exactly as each repo's train_ppo_fast.py env builds them,
sampled at one random prefix per case (seed 123), resources cycled 0..3 for
the primary target. Output: artifacts/generality/<target>.json

Requires the sb3 environment (see requirements.txt), NOT the repo's .venv:
    python experiments/generality_ppo/run_dual_level_ppo.py --lib <data_dir>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from dual_level import MarginHead, dual_level_study  # noqa: E402

DEFAULT_LIB = REPO.parents[0] / "libraries-prescriptive-process/01-rl-online"
OUT = REPO / "artifacts/generality"
SEED = 123
N_STATES = 500


def sample_rows(df: pd.DataFrame, case_col: str) -> pd.DataFrame:
    """One random prefix per case, up to N_STATES cases."""
    picked = df.groupby(case_col, sort=False).sample(n=1, random_state=SEED)
    if len(picked) > N_STATES:
        picked = picked.sample(n=N_STATES, random_state=SEED)
    return picked.reset_index(drop=True)


def states_rl_prescriptive(lib: Path) -> tuple[np.ndarray, list[str]]:
    csv = lib / "RL-prescriptive-monitoring/rl/data/ready_to_use_adaptive_bpic2017.csv"
    df = pd.read_csv(csv, sep=";")
    rows = sample_rows(df, "case_id")
    rel = (rows["prefix_nr"] / rows["case_length"].clip(lower=1)).clip(0, 1)
    resources = np.arange(len(rows)) % 4  # cycle 0..3 deterministically
    states = np.stack(
        [rel.to_numpy(), rows["reliability"].to_numpy(), rows["deviation"].to_numpy(), resources],
        axis=1,
    ).astype(np.float32)
    return states, ["relative_position", "reliability", "deviation", "available_resources"]


def states_when_to_treat(lib: Path) -> tuple[np.ndarray, list[str]]:
    csv = lib / "WhenToTreat/RL/data/results_adaptive_counterfacs_bpic2017.csv"
    df = pd.read_csv(csv)  # comma-separated, unlike the primary target
    rows = sample_rows(df, "Case ID")
    rel = (rows["event_nr"] / rows["case_length"].clip(lower=1)).clip(0, 1)
    states = np.stack(
        [rel.to_numpy(), rows["lower"].to_numpy(), rows["upper"].to_numpy()], axis=1
    ).astype(np.float32)
    return states, ["relative_position", "lower_TE", "upper_TE"]


TARGETS = {
    "rl-prescriptive-monitoring": (
        HERE / "models/ppo_bpic2017_rl_prescriptive_monitoring.zip",
        states_rl_prescriptive,
    ),
    "when-to-treat": (
        HERE / "models/ppo_bpic2017_when_to_treat.zip",
        states_when_to_treat,
    ),
}


def full_pool(name: str, lib: Path) -> tuple[np.ndarray, dict]:
    """Every prefix (resources cycling for the primary target), for the margin scan.

    The policies intervene rarely, so margin states are collected from the full
    prefix pool -- the analog of the paper's explained intervention cases --
    rather than from the one-prefix-per-case risk sample. The returned dict
    carries pool-level diagnostics (treatment-effect fractions from the repo's
    own counterfactual columns) so paper claims about the explanandum being
    vacuous trace to this artifact.
    """
    if name == "rl-prescriptive-monitoring":
        csv = lib / "RL-prescriptive-monitoring/rl/data/ready_to_use_adaptive_bpic2017.csv"
        df = pd.read_csv(csv, sep=";")
        rel = (df["prefix_nr"] / df["case_length"].clip(lower=1)).clip(0, 1)
        resources = np.arange(len(df)) % 4
        states = np.stack(
            [rel.to_numpy(), df["reliability"].to_numpy(), df["deviation"].to_numpy(), resources],
            axis=1,
        ).astype(np.float32)
        diag = {
            "frac_positive_te_pointwise": float(
                (df["Proba_if_Treated"] > df["Proba_if_Untreated"]).mean()
            ),
            "frac_positive_lower_cate": float((df["lower_cate"] > 0).mean()),
        }
        return states, diag
    csv = lib / "WhenToTreat/RL/data/results_adaptive_counterfacs_bpic2017.csv"
    df = pd.read_csv(csv)
    rel = (df["event_nr"] / df["case_length"].clip(lower=1)).clip(0, 1)
    states = np.stack(
        [rel.to_numpy(), df["lower"].to_numpy(), df["upper"].to_numpy()], axis=1
    ).astype(np.float32)
    return states, {"frac_positive_lower_te": float((df["lower"] > 0).mean())}


def pool_margin_stats(model, pool: np.ndarray, batch: int = 65536) -> dict:
    """Actor logit margin over the full pool: how often (and how strongly) the
    policy prefers intervening. Backs the 'never intervenes' verdict."""
    head = MarginHead(model.policy)
    outs = []
    with torch.no_grad():
        for i in range(0, len(pool), batch):
            outs.append(head(torch.from_numpy(pool[i : i + batch])).numpy())
    m = np.concatenate(outs)
    return {
        "mean": float(m.mean()),
        "max": float(m.max()),
        "frac_positive": float((m > 0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lib",
        type=Path,
        default=Path(os.environ.get("XPPM_PPO_LIB", DEFAULT_LIB)),
        help="Directory containing the two third-party repo checkouts with their state CSVs.",
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for name, (model_path, builder) in TARGETS.items():
        model = PPO.load(model_path, device="cpu")
        states, feature_names = builder(args.lib)
        expected = model.observation_space.shape[0]
        assert (
            states.shape[1] == expected
        ), f"{name}: state dim {states.shape[1]} != model obs dim {expected}"
        pool, pool_diag = full_pool(name, args.lib)
        result = dual_level_study(model, states, feature_names, seed=SEED, margin_pool=pool)
        result["model_path"] = str(model_path.relative_to(REPO))
        result["n_pool"] = int(len(pool))
        result["pool_diagnostics"] = pool_diag
        result["pool_margin"] = pool_margin_stats(model, pool)
        out_path = OUT / f"{name}.json"
        out_path.write_text(json.dumps(result, indent=1))
        print(
            f"[{name}] n={result['n_states']} margin_n={result['n_margin_states']} "
            f"wait_n={result['n_wait_states']} "
            f"pool_intervene_rate={result['intervene_rate']:.4f} -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
