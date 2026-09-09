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
    reliability = deviation                        (BPIC2012: ensemble of one)
                = max(r, 1 - r)                    (BPIC2017: prob. of the predicted class)
    y1 = 1[p_T > 0.5],  y0 = 1[p_U > 0.5]         (Y = 1 the undesired outcome)
    treatment   = the case's recorded treatment flag (historical action)

The split is the same temporal test split both retrained models were
evaluated on (risk_model.temporal_split), i.e. out-of-sample scores, as in
the released pipeline; on BPIC2012 it is row-for-row the shipped CSV's case
set. Writes paths.retrained_csv(log) (git-ignored like the shipped CSVs).

Usage: python build_retrained_state.py [--logs BPIC2012 BPIC2017]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import compose as cp
import effect_model as em
import paths
import risk_model as rm

LOGS = ("BPIC2012", "BPIC2017")


def build(log: str) -> pd.DataFrame:
    df, conf = rm.load_events(log)
    _tr, te, _va = rm.temporal_split(df, conf)
    X, meta = rm.encode_prefixes(te, conf)
    clf, rfeats = rm.load_model(log)
    arms, efeats = em.load_model(log)
    r = clf.predict_proba(X[rfeats["feature_names"]])[:, 1]
    Xd = em.one_hot(X[efeats["raw_columns"]], efeats["cat_cols"], efeats["columns"])
    pT, pU = arms["treated"].predict_proba(Xd)[:, 1], arms["untreated"].predict_proba(Xd)[:, 1]
    reliability, deviation = cp.risk_features_from_r(log, r)
    out = pd.DataFrame({
        "case_id": meta["case_id"].to_numpy(), "prefix_nr": meta["prefix_nr"].to_numpy(), "case_length": meta["case_length"].to_numpy(),
        "orig_timestamp": meta["timestamp"].to_numpy(), "actual": meta["y"].to_numpy(),
        "predicted": (r > 0.5).astype(int), "predicted_proba_0": 1.0 - r, "predicted_proba_1": r,
        "reliability": reliability, "deviation": deviation,
        "Proba_if_Treated": pT, "Proba_if_Untreated": pU, "y1": (pT > 0.5).astype(int), "y0": (pU > 0.5).astype(int),
        "treatment": meta["t"].to_numpy(),
    })
    return out.sort_values(["orig_timestamp", "prefix_nr"], kind="mergesort").reset_index(drop=True)


def compare_with_shipped(log: str, new: pd.DataFrame) -> dict:
    shipped = pd.read_csv(paths.bpic_csv(log, "released"), sep=";", usecols=["case_id", "prefix_nr", "predicted_proba_1", "deviation", "Proba_if_Treated", "Proba_if_Untreated", "y1", "y0"])
    shipped["case_id"] = shipped["case_id"].astype(str)
    j = shipped.merge(new, on=["case_id", "prefix_nr"], suffixes=("_s", "_n"))
    if not len(j):
        return {"overlap_rows": 0}
    return {"overlap_rows": int(len(j)), "corr_r": float(np.corrcoef(j.predicted_proba_1_s, j.predicted_proba_1_n)[0, 1]),
            "deviation_agreement": float((j.deviation_s == j.deviation_n).mean()),
            "corr_pT": float(np.corrcoef(j.Proba_if_Treated_s, j.Proba_if_Treated_n)[0, 1]),
            "positive_rule_agreement": float(((j.y1_s - j.y0_s > 0) == (j.y1_n - j.y0_n > 0)).mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", default=list(LOGS), choices=list(LOGS))
    a = ap.parse_args()
    for lg in a.logs:
        out = build(lg)
        p = paths.retrained_csv(lg)
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, sep=";", index=False)
        ite = out.y1 - out.y0
        print(f"{lg}: {len(out)} prefixes of {out.case_id.nunique()} cases -> {p}")
        print(f"  r mean {out.predicted_proba_1.mean():.3f}, predicted deviant {out.predicted.mean():.3f}, actual deviant {out.actual.mean():.3f}; "
              f"p_T mean {out.Proba_if_Treated.mean():.3f}, p_U mean {out.Proba_if_Untreated.mean():.3f}, ite>0 share {(ite > 0).mean():.3f}, treated cases {out.treatment.mean():.3f}")
        print("  vs shipped:", compare_with_shipped(lg, out))
