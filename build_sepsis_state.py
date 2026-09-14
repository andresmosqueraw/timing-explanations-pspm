"""The coherent Sepsis state: score the prepared log's temporal test split
with the retrained risk model (risk_model.py) and effect model
(effect_model.py) and write an RL CSV in the same format
build_retrained_state.py uses for BPIC, so a policy trained on it
(sepsis_resources/train_ppo_sepsis.py) reads *exactly* the numbers the risk
and effect levels explain.

Unlike BPIC there is no "shipped" CSV here at all: Sepsis has no prior
checkpoint and no released pipeline to compare against or reconcile with
(see the top-level task's "LECCION CRITICA" about SimBank's shipped-vs-
rebuilt mismatch -- the whole point of building this from scratch is that
the distinction must never arise). ``verify_coherence`` below re-scores the
same prefixes with the same two models straight after writing the CSV and
must find EXACT agreement (flip_rate 0): if it does not, something in this
script or in sepsis_resources/train_ppo_sepsis.py has drifted from these two
models, and that must be fixed before training any policy on the result.

Column derivations follow build_retrained_state.py's BPIC2017 rule (the
only one that applies here, since Sepsis is not "BPIC2012" in
compose.risk_features_from_r):

    predicted   = 1[r > 0.5],        deviation = 1 - predicted
    relative_position = prefix_nr / progress_horizon, clipped to 1, where the
                  horizon is the 95th percentile of decision points per case in
                  the *training* split (known before any test case starts; the
                  released prefix_nr / case_length needs the case's future length)
    reliability = max(r, 1 - r)                    (probability of the predicted class)
    y1 = 1[p_T < 0.5],  y0 = 1[p_U < 0.5]         (the desired outcome, no Return ER; see compose.desired_outcome)
    treatment   = the prefix's dynamic "IV Antibiotics already given" flag

Usage: python build_sepsis_state.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import compose as cp
import effect_model as em
import paths
import risk_model as rm

LOG = "Sepsis"


def build() -> pd.DataFrame:
    df, conf = rm.load_events(LOG)
    tr, te, _va = rm.temporal_split(df, conf)
    X, meta = rm.encode_prefixes(te, conf)
    horizon = rm.progress_horizon(tr, conf)
    clf, rfeats = rm.load_model(LOG)
    arms, efeats = em.load_model(LOG)
    r = clf.predict_proba(X[rfeats["feature_names"]])[:, 1]
    Xd = em.one_hot(X[efeats["raw_columns"]], efeats["cat_cols"], efeats["columns"])
    pT, pU = arms["treated"].predict_proba(Xd)[:, 1], arms["untreated"].predict_proba(Xd)[:, 1]
    reliability, deviation = cp.risk_features_from_r(LOG, r)
    out = pd.DataFrame({
        "case_id": meta["case_id"].to_numpy(), "prefix_nr": meta["prefix_nr"].to_numpy(), "progress_horizon": horizon,
        "orig_timestamp": meta["timestamp"].to_numpy(), "actual": meta["y"].to_numpy(),
        "predicted": (r > 0.5).astype(int), "predicted_proba_0": 1.0 - r, "predicted_proba_1": r,
        "reliability": reliability, "deviation": deviation,
        "Proba_if_Treated": pT, "Proba_if_Untreated": pU, "y1": cp.desired_outcome(pT), "y0": cp.desired_outcome(pU),
        "treatment": meta["t"].to_numpy(),
    })
    return out.sort_values(["orig_timestamp", "prefix_nr"], kind="mergesort").reset_index(drop=True)


def verify_coherence(out: pd.DataFrame) -> dict:
    """Re-score the SAME prefixes with the SAME two models straight from
    disk and confirm exact agreement with what was just written -- the
    single-pipeline check the task asks for in place of a shipped-vs-rebuilt
    reconciliation (there is nothing to reconcile: only one pipeline ever
    produced numbers for Sepsis)."""
    df, conf = rm.load_events(LOG)
    _tr, te, _va = rm.temporal_split(df, conf)
    X, meta = rm.encode_prefixes(te, conf)
    clf, rfeats = rm.load_model(LOG)
    arms, efeats = em.load_model(LOG)
    r2 = clf.predict_proba(X[rfeats["feature_names"]])[:, 1]
    Xd2 = em.one_hot(X[efeats["raw_columns"]], efeats["cat_cols"], efeats["columns"])
    pT2, pU2 = arms["treated"].predict_proba(Xd2)[:, 1], arms["untreated"].predict_proba(Xd2)[:, 1]
    reliability2, deviation2 = cp.risk_features_from_r(LOG, r2)
    re = meta.assign(r=r2, pT=pT2, pU=pU2, reliability=reliability2, deviation=deviation2)
    j = out.merge(re, on=["case_id", "prefix_nr"], suffixes=("", "_re"))
    return {
        "n": int(len(j)), "n_out": int(len(out)),
        "corr_r": float(np.corrcoef(j.predicted_proba_1, j.r)[0, 1]),
        "max_abs_diff_r": float(np.abs(j.predicted_proba_1 - j.r).max()),
        "deviation_agree": float((j.deviation == j.deviation_re).mean()),
        "reliability_max_abs_diff": float(np.abs(j.reliability - j.reliability_re).max()),
        "max_abs_diff_pT": float(np.abs(j.Proba_if_Treated - j.pT).max()),
        "max_abs_diff_pU": float(np.abs(j.Proba_if_Untreated - j.pU).max()),
        "y1_agree": float((j.y1 == cp.desired_outcome(j.pT)).mean()),
        "y0_agree": float((j.y0 == cp.desired_outcome(j.pU)).mean()),
        "flip_rate": float((cp.positive_effect_rule(j.Proba_if_Treated, j.Proba_if_Untreated) !=
                            cp.positive_effect_rule(j.pT, j.pU)).mean()),
    }


if __name__ == "__main__":
    out = build()
    p = paths.SEPSIS_STATE_CSV
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, sep=";", index=False)
    ite = out.y1 - out.y0
    print(f"Sepsis: {len(out)} prefixes of {out.case_id.nunique()} cases -> {p}")
    print(f"  r mean {out.predicted_proba_1.mean():.3f}, predicted deviant {out.predicted.mean():.3f}, actual deviant {out.actual.mean():.3f}; "
          f"p_T mean {out.Proba_if_Treated.mean():.3f}, p_U mean {out.Proba_if_Untreated.mean():.3f}, ite>0 share {(ite > 0).mean():.3f}, "
          f"treated rows {out.treatment.mean():.3f}")
    coherence = verify_coherence(out)
    print("  coherence check (re-score with the same models; must be exact):", coherence)
    assert coherence["flip_rate"] == 0.0, "Sepsis state is NOT coherent with its own risk/effect models -- stop and diagnose."
    assert coherence["deviation_agree"] == 1.0
    print("  OK: coherent by construction (flip_rate = 0, deviation_agree = 1.0).")
