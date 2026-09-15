"""The coherent pipeline: score the prepared log's test split with the
retrained outcome predictor (risk_model.py) and causal-effect estimator
(effect_model.py) and write an RL CSV in Shoush & Dumas's format, so that a
policy trained on it (``pools.VARIANTS["cate_retrained"]``) reads *exactly*
the numbers the risk and effect levels explain.

The shipped RL CSVs carry the outputs of the pipeline's own run of those two
models; on BPIC2017 that run cannot be reproduced (the CSV comes from a
different predictor run than the prepared log, see README) and the retrained
models disagree with it (corr 0.02 on r, 0.07 on p_T over the evaluation
pool), so a composition read on the shipped state explains numbers the
policy did not see. This script removes the gap at the source. Column
derivations follow the released files exactly (verified on both CSVs):

    predicted   = 1[r > 0.5],        deviation = 1 - predicted
    relative_position = prefix_nr / progress_horizon, clipped to 1, where the
                  horizon is the 95th percentile of decision points per case in
                  the *training* split (known before any test case starts; the
                  released prefix_nr / case_length needs the case's future length)
    reliability = deviation                        (BPIC2012: ensemble of one)
                = max(r, 1 - r)                    (BPIC2017: prob. of the predicted class)
    y1 = 1[p_T < 0.5],  y0 = 1[p_U < 0.5]         (the *desired* outcome; p = P(undesired),
                                                   compose.desired_outcome: Shoush & Dumas's
                                                   released code thresholds p > 0.5, which pays
                                                   for treating where treatment causes the bad outcome)
    treatment   = the case's recorded treatment flag (historical action)

Splits and roles (the train / validation / test protocol, crossfit.py):

    train  scored out of fold (5 folds by case)  -> paths.retrained_csv(log, "train"): the agent trains here only
    val    scored by the final models            -> paths.retrained_csv(log, "val"): choosing the agent
    test   scored by the final models            -> paths.retrained_csv(log): final gain and every explanation

The risk and effect models are fit on the training split (the risk model
stops early on the validation split); the test split is used only at the
end. Each row carries its ``fold`` (-1 outside train); the fold record goes
to crossfit_manifest.json. On BPIC2012 the test CSV is row-for-row the
shipped CSV's case set. The CSVs are git-ignored like the shipped ones.

Usage: python build_retrained_state.py [--logs BPIC2012 BPIC2017 Sepsis] [--splits train val test]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import compose as cp
import crossfit
import paths
import risk_model as rm

LOGS = ("BPIC2012", "BPIC2017", "Sepsis")


def build(log: str, split: str = "test") -> tuple[pd.DataFrame, dict]:
    X, meta, r, pT, pU, fold, tr, info = crossfit.score_split(log, split)
    _, conf = rm.load_events(log)
    horizon = rm.progress_horizon(tr, conf)
    reliability, deviation = cp.risk_features_from_r(log, r)
    out = pd.DataFrame({
        "case_id": meta["case_id"].to_numpy(), "prefix_nr": meta["prefix_nr"].to_numpy(), "progress_horizon": horizon,
        "orig_timestamp": meta["timestamp"].to_numpy(), "actual": meta["y"].to_numpy(),
        "predicted": (r > 0.5).astype(int), "predicted_proba_0": 1.0 - r, "predicted_proba_1": r,
        "reliability": reliability, "deviation": deviation,
        "Proba_if_Treated": pT, "Proba_if_Untreated": pU, "y1": cp.desired_outcome(pT), "y0": cp.desired_outcome(pU),
        "treatment": meta["t"].to_numpy(), "fold": fold,
    })
    return out.sort_values(["orig_timestamp", "prefix_nr"], kind="mergesort").reset_index(drop=True), info


def compare_with_shipped(log: str, new: pd.DataFrame) -> dict:
    shipped = pd.read_csv(paths.bpic_csv(log, "released"), sep=";", usecols=["case_id", "prefix_nr", "predicted_proba_1", "deviation", "Proba_if_Treated", "Proba_if_Untreated"])
    shipped["case_id"] = shipped["case_id"].astype(str)
    j = shipped.merge(new, on=["case_id", "prefix_nr"], suffixes=("_s", "_n"))
    if not len(j):
        return {"overlap_rows": 0}
    return {"overlap_rows": int(len(j)), "corr_r": float(np.corrcoef(j.predicted_proba_1_s, j.predicted_proba_1_n)[0, 1]),
            "deviation_agreement": float((j.deviation_s == j.deviation_n).mean()),
            "corr_pT": float(np.corrcoef(j.Proba_if_Treated_s, j.Proba_if_Treated_n)[0, 1]),
            "positive_rule_agreement": float((cp.positive_effect_rule(j.Proba_if_Treated_s, j.Proba_if_Untreated_s) == cp.positive_effect_rule(j.Proba_if_Treated_n, j.Proba_if_Untreated_n)).mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", default=list(LOGS), choices=list(LOGS))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    a = ap.parse_args()
    vs_path = paths.REPO / "retrained_state_vs_shipped.json"
    vs = json.loads(vs_path.read_text()) if vs_path.exists() else {}
    cf = json.loads(paths.CROSSFIT_JSON.read_text()) if paths.CROSSFIT_JSON.exists() else {}
    for lg in a.logs:
        for split in a.splits:
            out, info = build(lg, split)
            p = paths.retrained_csv(lg, split)
            p.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(p, sep=";", index=False)
            ite = out.y1 - out.y0
            print(f"{lg} [{split}]: {len(out)} prefixes of {out.case_id.nunique()} cases -> {p}")
            print(f"  r mean {out.predicted_proba_1.mean():.3f}, predicted deviant {out.predicted.mean():.3f}, actual deviant {out.actual.mean():.3f}; "
                  f"p_T mean {out.Proba_if_Treated.mean():.3f}, p_U mean {out.Proba_if_Untreated.mean():.3f}, ite>0 share {(ite > 0).mean():.3f}, treated {out.treatment.mean():.3f}", flush=True)
            if split == "train":
                cf[lg] = {**info, "n_prefixes": int(len(out)), "n_cases": int(out.case_id.nunique())}
                paths.CROSSFIT_JSON.write_text(json.dumps(cf, indent=2))
            if split == "test" and lg != "Sepsis":
                vs[lg] = compare_with_shipped(lg, out)
                print("  vs shipped:", vs[lg])
    vs_path.write_text(json.dumps(vs, indent=2))
