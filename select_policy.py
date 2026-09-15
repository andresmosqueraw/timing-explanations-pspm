"""Choose each log's agent on the validation split, never on test.

Every candidate of the sweep (models/variants/sweep_oof/, trained on the
out-of-fold *training* state only, see crossfit.py) is scored by its mean
reward per decision point on the validation cases. The best one becomes the
paper's checkpoint (paths.variant_model(log, pools.DEFAULT_VARIANT)) and the
table goes to policy_selection.json. The test split is not read here.

Usage: python select_policy.py [--logs BPIC2012 BPIC2017 Sepsis]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys

import numpy as np
from stable_baselines3 import PPO

import paths
import pools

sys.path.insert(0, str(paths.REPO / "foreign"))
sys.path.insert(0, str(paths.REPO / "sepsis_resources"))
from train_ppo_fast_rl_prescriptive_monitoring import PPMEnvFast  # noqa: E402
from train_ppo_sepsis import SepsisEnv  # noqa: E402

SWEEP = paths.VARIANT_MODELS / "sweep_oof"
VAL_SAMPLE = 30000  # BPIC2017's validation split has >200k decision points; same sample size as the gain table


def val_reward(log: str, model_path) -> dict:
    model = PPO.load(str(model_path), device="cpu")
    if log == "Sepsis":
        states, ite, _ = pools.sepsis_full(paths.retrained_csv(log, "val"))
        act = model.predict(states, deterministic=True)[0].astype(int)
        r = np.array([SepsisEnv._reward(bool(a), v) for a, v in zip(act, ite)])
    else:
        states, ite, _ = pools.bpic_full(paths.retrained_csv(log, "val"), pools.treatment_col(log, pools.DEFAULT_VARIANT),
                                         sample=VAL_SAMPLE if log == "BPIC2017" else None)
        act = model.predict(states, deterministic=True)[0].astype(int)
        pool = [1] * pools.N_RESOURCES
        r = np.array([PPMEnvFast._reward(bool(a), v, pool) for a, v in zip(act, ite)])
    return {"val_mean_reward": float(r.mean()), "val_intervene_rate": float(act.mean()), "n_val": int(len(r))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", default=["BPIC2012", "BPIC2017", "Sepsis"])
    a = ap.parse_args(argv)
    sel = json.loads(paths.POLICY_SELECTION_JSON.read_text()) if paths.POLICY_SELECTION_JSON.exists() else {}
    for log in a.logs:
        rows = []
        # SB3 saves "ppo_<log>_ent0.3" without ".zip" (it reads ".3" as an extension)
        for z in sorted(SWEEP.glob(f"ppo_{log.lower()}_ent*")):
            m = re.fullmatch(rf"ppo_{log.lower()}_ent([0-9.]+?)(?:_(\d+)k)?(?:\.zip)?", z.name)
            if not m:
                continue
            base = SWEEP / f"ppo_{log.lower()}_ent{m.group(1)}"
            manifest = json.loads(base.with_name(base.name + "_manifest.json").read_text())
            steps = int(m.group(2)) * 1000 if m.group(2) else int(manifest["total_timesteps"])
            row = {"checkpoint": z.name, "ent_coef": float(m.group(1)), "timesteps": steps, **val_reward(log, z)}
            rows.append(row)
            print(f"{log}: {row}", flush=True)
        if not rows:
            print(f"{log}: no candidates in {SWEEP}")
            continue
        best = max(rows, key=lambda r: r["val_mean_reward"])
        dst = paths.variant_model(log, pools.DEFAULT_VARIANT)
        shutil.copyfile(SWEEP / best["checkpoint"], dst)
        base = SWEEP / f"ppo_{log.lower()}_ent{best['ent_coef']:g}"
        manifest = json.loads(base.with_name(base.name + "_manifest.json").read_text())
        manifest.update({"total_timesteps": best["timesteps"], "selected_on": "validation split, mean reward per decision point",
                         "selection": best, "trained_on": "training split, out-of-fold state (crossfit.py)"})
        manifest["args"]["timesteps"] = best["timesteps"]
        dst.with_name(dst.stem + "_manifest.json").write_text(json.dumps(manifest, indent=2))
        for suffix in ("_training_curve.csv", "_monitor.csv"):
            src = base.with_name(base.name + suffix)
            if src.exists():
                shutil.copyfile(src, dst.with_name(dst.stem + suffix))
        sel[log] = {"candidates": rows, "chosen": best, "checkpoint": str(dst.relative_to(paths.REPO))}
        print(f"{log}: chosen {best['checkpoint']} -> {dst}")
        paths.POLICY_SELECTION_JSON.write_text(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
