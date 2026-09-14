"""Leakage audit of the coherent pipeline: no information from a case's future
reaches a decision point, its risk and effect models, or the policy's state.

Per log, recomputed from the raw events rather than from the code it audits:

  1. decision points: every prefix the models are trained or scored on ends
     before the case's first outcome-revealing event and before the
     intervention (risk_model.add_decision_points), and every later prefix is
     excluded;
  2. features: none of the columns known to carry the future
     (risk_model.LEAK_NOTE) is an input of the risk or the effect model, and
     static attributes take only the value known so far in the prefix;
  3. scan: no single attribute condition observable in a test prefix makes
     the outcome (near-)certain, and no single encoded feature separates the
     outcome with a test AUC above ``AUC_FLAG`` (each feature's train-split
     mapping scored on the test split);
  4. treatment: the propensity's overlap on the test decision points (the
     effect estimator's positivity);
  5. state: the RL CSV holds exactly the test decision points, carries no
     case length, and its relative_position uses the fixed training horizon.

Writes leakage_audit.json; exits non-zero when a hard check fails.

Usage: python leakage_audit.py [--logs BPIC2012 BPIC2017 Sepsis]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import effect_model as em
import paths
import risk_model as rm

LOGS = ("BPIC2012", "BPIC2017", "Sepsis")
FORBIDDEN = {"time_to_event_m", "NumberOfOffers", "CreditScore", "Accepted", "Selected", "case_length"}
STATE_CSV = {"BPIC2012": paths.retrained_csv("BPIC2012"), "BPIC2017": paths.retrained_csv("BPIC2017"), "Sepsis": paths.SEPSIS_STATE_CSV}
NEAR_CERTAIN = 0.01   # a condition whose outcome rate is below this or above 1 - this
MIN_SUPPORT = 200     # ... on at least this many test prefixes, is a leak
AUC_FLAG = 0.90       # a single feature separating the outcome better than this is flagged


def _base(name: str) -> str:
    return name.rsplit("_", 1)[0] if name.endswith(("_mean", "_max", "_min", "_sum", "_std")) else name


def _event_counts(df: pd.DataFrame, conf: dict) -> pd.DataFrame:
    """Per event, in encode_prefixes' order: prefix number and how many
    outcome-revealing / treatment events the prefix ending there contains."""
    case, ts, act = conf["case_col"], conf["ts_col"], conf["activity_col"]
    d = df.sort_values([case, ts, act], kind="mergesort")
    g = d.groupby(case, sort=False)
    return pd.DataFrame({"case_id": d[case].astype(str).to_numpy(), "prefix_nr": (g.cumcount() + 1).to_numpy(),
                         "n_outcome": d[act].isin(conf["outcome_activities"]).groupby(d[case], sort=False).cumsum().to_numpy(),
                         "n_treat": (d[act] == conf["treatment_activity"]).groupby(d[case], sort=False).cumsum().to_numpy(),
                         "decision": d["_decision"].to_numpy(bool)}, index=d.index)


def audit(log: str) -> dict:
    df, conf = rm.load_events(log)
    case, ts, act = conf["case_col"], conf["ts_col"], conf["activity_col"]
    out: dict = {"log": log, "checks": {}}
    chk = out["checks"]

    # 1. decision points, on every event of the log
    ev = _event_counts(df, conf)
    before = (ev.n_outcome == 0) & (ev.n_treat < conf["treatment_nth"])
    chk["decision_points_precede_outcome_and_treatment"] = bool((~ev.decision | before).all())
    chk["later_prefixes_excluded"] = bool((ev.decision | ~before).all())
    out["decision_points"] = {"events": int(len(ev)), "decision_points": int(ev.decision.sum())}

    tr, te, _va = rm.temporal_split(df, conf)
    Xtr, mtr = rm.encode_prefixes(tr, conf)
    Xte, mte = rm.encode_prefixes(te, conf)
    keys = ev.loc[ev.decision, ["case_id", "prefix_nr"]]
    chk["encoded_prefixes_are_decision_points"] = bool(len(mte.merge(keys, on=["case_id", "prefix_nr"])) == len(mte))

    # 2. features
    rfeats = json.loads(rm.model_paths(log)["features"].read_text())["feature_names"]
    efeats = json.loads(em.model_paths(log)["features"].read_text())["raw_columns"]
    used = {_base(f) for f in rfeats} | {_base(f) for f in efeats}
    chk["no_forbidden_model_inputs"] = not (used & FORBIDDEN)
    out["model_inputs"] = sorted(used)
    d = te.sort_values([case, ts, act], kind="mergesort")
    g = d.groupby(case, sort=False)
    static_ok = True
    for c in conf["static_num"] + conf["static_cat"]:
        known = g[c].ffill()
        known = known.fillna(0.0).astype(float) if c in conf["static_num"] else known.fillna("0").astype(str)
        ref = pd.DataFrame({"case_id": d[case].astype(str).to_numpy(), "prefix_nr": (g.cumcount() + 1).to_numpy(), "known": known.to_numpy()})
        j = mte[["case_id", "prefix_nr"]].assign(x=Xte[c].to_numpy()).merge(ref, on=["case_id", "prefix_nr"])
        static_ok &= bool((j.x == j.known).all())
    chk["static_attributes_known_so_far"] = static_ok

    # 3. scan on the test decision prefixes
    y = (d[conf["label_col"]] == conf["pos_label"]).astype(int).to_numpy()
    dec = d["_decision"].to_numpy(bool)
    yd, base = y[dec], y[dec].mean()
    conds = []
    for c in conf["dynamic_num"] + conf["static_num"]:
        seen = (d[c].fillna(0) != 0).groupby(d[case], sort=False).cummax().to_numpy()[dec]
        for flag, m in (("seen != 0", seen), ("never != 0", ~seen)):
            if m.sum() >= MIN_SUPPORT:
                conds.append((f"{c} {flag}", int(m.sum()), float(yd[m].mean())))
    for c in conf["dynamic_cat"] + conf["static_cat"]:
        s = d[c].fillna("0").astype(str)
        for v in s[dec].value_counts().index:
            m = (s == v).groupby(d[case], sort=False).cummax().to_numpy()[dec]
            if m.sum() >= MIN_SUPPORT:
                conds.append((f"{c} == {v!r} seen", int(m.sum()), float(yd[m].mean())))
    flagged = [c for c in conds if c[2] <= NEAR_CERTAIN or c[2] >= 1 - NEAR_CERTAIN]
    chk["no_near_certain_condition"] = not flagged
    out["scan"] = {"base_rate": float(base), "n_conditions": len(conds), "near_certain": flagged,
                   "most_extreme": sorted(conds, key=lambda c: -abs(c[2] - base))[:10]}

    ytr, yte = mtr["y"].to_numpy(), mte["y"].to_numpy()
    aucs = {}
    for c in Xte.columns:
        if Xte[c].dtype == object:
            rate = pd.Series(ytr).groupby(Xtr[c].to_numpy()).mean()
            score = Xte[c].map(rate).fillna(ytr.mean()).to_numpy(float)
        else:
            score = Xte[c].to_numpy(float)
        if np.unique(score).size > 1:
            a = roc_auc_score(yte, score)
            aucs[c] = float(max(a, 1 - a))
    top = sorted(aucs.items(), key=lambda kv: -kv[1])[:10]
    chk["no_single_feature_above_auc_flag"] = top[0][1] <= AUC_FLAG
    out["single_feature_auc_top10"] = top
    risk_m = json.loads(rm.model_paths(log)["manifest"].read_text())
    out["risk_model_test_auc"] = risk_m["test_auc"]

    # 4. positivity of the effect estimator
    prop = pickle.loads(em.model_paths(log)["propensity"].read_bytes())
    ef = json.loads(em.model_paths(log)["features"].read_text())
    Xd = em.one_hot(Xte[ef["raw_columns"]], ef["cat_cols"], ef["columns"])
    e = prop.predict_proba(Xd.to_numpy(dtype=np.float32))[:, 1]
    t = mte["t"].to_numpy()
    out["propensity"] = {"treated_share_test": float(t.mean()), "auc": float(roc_auc_score(t, e)) if np.unique(t).size > 1 else None,
                         "share_in_0.05_0.95": float(((e >= 0.05) & (e <= 0.95)).mean()), "min": float(e.min()), "max": float(e.max()),
                         "cases_in_overlap": int(mte.loc[(e >= 0.05) & (e <= 0.95), "case_id"].nunique()), "cases": int(mte.case_id.nunique())}
    # not a leak but a limit of the effect level: without overlap the two arms are never compared on similar cases
    out["warnings"] = [] if out["propensity"]["share_in_0.05_0.95"] >= 0.5 else [
        f"weak positivity: only {out['propensity']['share_in_0.05_0.95']:.1%} of test decision points have a propensity in [0.05, 0.95]"]

    # 5. the RL state
    p = STATE_CSV[log]
    if p.exists():
        st = pd.read_csv(p, sep=";", dtype={"case_id": str}, keep_default_na=False, na_values=[])
        chk["state_has_no_case_length"] = "case_length" not in st.columns
        k = st[["case_id", "prefix_nr"]].merge(mte[["case_id", "prefix_nr"]], on=["case_id", "prefix_nr"])
        chk["state_rows_are_test_decision_points"] = bool(len(k) == len(st) == len(mte))
        chk["state_horizon_from_training"] = bool(np.allclose(st["progress_horizon"], rm.progress_horizon(tr, conf)))
    ok = all(chk.values())
    print(f"\n=== {log}: {'PASS' if ok else 'FAIL'}")
    for name, v in chk.items():
        print(f"  [{'ok' if v else 'FAIL'}] {name}")
    print(f"  base rate {base:.3f}; most extreme condition {out['scan']['most_extreme'][0]}")
    print(f"  top single-feature AUC {top[:3]}; risk model test AUC {out['risk_model_test_auc']:.3f}")
    print(f"  propensity {out['propensity']}")
    out["pass"] = ok
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", default=list(LOGS), choices=list(LOGS))
    a = ap.parse_args(argv)
    res = [audit(lg) for lg in a.logs]
    (paths.REPO / "leakage_audit.json").write_text(json.dumps(res, indent=2, default=float))
    print("\nSaved -> leakage_audit.json")
    sys.exit(0 if all(r["pass"] for r in res) else 1)


if __name__ == "__main__":
    main()
