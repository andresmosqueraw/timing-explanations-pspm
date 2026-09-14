"""The causal-effect estimator behind the state's treatment-effect features,
retrained -- the third model whose numbers the policy reads.

The policy's ``Proba_if_Treated`` / ``Proba_if_Untreated`` features are not
log attributes either: Shoush & Dumas's offline phase
(``causal/causallift_adapted.py``) fits CausalLift's *two-model* estimator
(a T-learner) on the same static + aggregate prefix encoding the outcome
predictor uses:

    propensity  e(x) = P(T = 1 | x)          LogisticRegression (grid over C
                                             in {0.1, 1, 10}, l1/l2, liblinear,
                                             3-fold CV), for inverse-probability
                                             weights w = T/e(x) + (1-T)/(1-e(x))
    treated arm  p_T(x) = P(Y = 1 | T=1, x)  XGBClassifier(max_depth=3,
    untreated    p_U(x) = P(Y = 1 | T=0, x)  n_estimators=100, learning_rate=0.1,
                                             random_state=0), each fitted on its
                                             arm's rows with weights w
    CATE(x) = p_U(x) - p_T(x)   (the drop in P(undesired) the treatment buys);
    the reward's y1 = 1[p_T < 0.5], y0 = 1[p_U < 0.5]  (compose.desired_outcome)

with Y = 1 the *deviant* (undesired) outcome and T the case's recorded
treatment. This module retrains that estimator per BPIC log with the same
split and encoding (``risk_model``), so that p_T, p_U and their difference
can be attributed with TreeSHAP -- an *effect explanation*: why does the
estimator expect the intervention to change this case's outcome? SimBank
has no such estimator (its effect feature is the simulator's own quality
uncertainty), so it has no effect model.

Faithfulness to the released pipeline: same features, same learners, same
hyper-parameters; categorical attributes are one-hot encoded (XGBoost needs
numeric input), and the released ``time_to_event_m`` column is dropped for
the reason given in ``risk_model.LEAK_NOTE``. The manifest records the
correlation of the retrained p_T / p_U with the shipped ones on the rows
the two pipelines share.

Usage:
    python effect_model.py                # BPIC2012 and BPIC2017 -> models/effect/
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import compose as cp
import paths
import risk_model as rm

EFFECT_LOGS = ("BPIC2012", "BPIC2017", "SimBank", "Sepsis")
# Sepsis: t is the dynamic "IV Antibiotics already given" flag built in
# risk_model.load_sepsis_events, so effect_rows below (default branch, no
# SimBank-style restriction) fits the T-learner on every prefix of every
# case, both arms present throughout -- unlike SimBank there is no single
# fixed decision event to restrict to.
# SimBank: the log's "normal" policy contacts HQ, when it does, always at
# event 5 (after the first customer contact) and skips at event 9 or 10, so
# the only prefixes on which both arms exist are the first four events; the
# estimator is fit there (prefix_nr <= SIMBANK_COMMON_PREFIX and before the
# case's decision) and applied to every event of the log afterwards.
SIMBANK_COMMON_PREFIX = 4


def effect_rows(log: str, meta: pd.DataFrame) -> np.ndarray:
    """Boolean mask of the prefixes the estimator is fit and scored on."""
    if log != "SimBank":
        return np.ones(len(meta), bool)
    dp = meta["decision_prefix"].to_numpy()
    return np.isfinite(dp) & (meta["prefix_nr"].to_numpy() < dp) & (meta["prefix_nr"].to_numpy() <= SIMBANK_COMMON_PREFIX)
XGB_PARAMS = dict(max_depth=3, n_estimators=100, learning_rate=0.1, objective="binary:logistic",
                  random_state=0, n_jobs=-1, tree_method="hist")


class ConstantArm:
    """An arm whose training outcome has a single class (SimBank: every case
    that skips the HQ contact is cancelled), so its probability is a constant
    and its attribution is zero."""

    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


def model_paths(log: str) -> dict:
    d = paths.RISK_MODELS.parent / "effect"
    return {"dir": d, "treated": d / f"{log}_xgb_treated.json", "untreated": d / f"{log}_xgb_untreated.json",
            "const_treated": d / f"{log}_const_treated.json", "const_untreated": d / f"{log}_const_untreated.json",
            "propensity": d / f"{log}_propensity.pkl", "features": d / f"{log}_features.json", "manifest": d / f"{log}_manifest.json"}


def one_hot(X: pd.DataFrame, cat_cols: list[str], columns: list[str] | None = None) -> pd.DataFrame:
    Xd = pd.get_dummies(X, columns=cat_cols, dtype=float)
    if columns is not None:
        Xd = Xd.reindex(columns=columns, fill_value=0.0)
    return Xd


def fit_propensity(X: pd.DataFrame, t: np.ndarray, seed: int = 0, n_jobs: int = 4):
    """CausalLift's propensity grid. The design matrix is passed as a float32
    numpy array so joblib memory-maps one shared copy across the workers
    (a DataFrame is pickled to every worker: 12 x 3 GB on BPIC2017), and the
    grid runs on ``n_jobs`` workers rather than every core."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    Xa = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
    grid = GridSearchCV(LogisticRegression(solver="liblinear", max_iter=1000, random_state=seed),
                        {"C": [0.1, 1, 10], "penalty": ["l1", "l2"]}, cv=3, n_jobs=n_jobs, pre_dispatch="n_jobs")
    grid.fit(Xa, t)
    return grid.best_estimator_


def ipw_weights(e: np.ndarray, t: np.ndarray, clip: float = 1e-3) -> np.ndarray:
    e = np.clip(e, clip, 1 - clip)
    return np.where(t == 1, 1.0 / e, 1.0 / (1.0 - e))


def train(log: str, seed: int = 0, max_train_rows: int | None = None) -> dict:
    import pickle

    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    t0 = time.time()
    df, conf = rm.load_events(log)
    tr, te, va = rm.temporal_split(df, conf)
    Xtr, mtr = rm.encode_prefixes(tr, conf)
    Xte, mte = rm.encode_prefixes(te, conf)
    ktr, kte = effect_rows(log, mtr), effect_rows(log, mte)
    Xtr, mtr, Xte, mte = Xtr[ktr].reset_index(drop=True), mtr[ktr].reset_index(drop=True), Xte[kte].reset_index(drop=True), mte[kte].reset_index(drop=True)
    if max_train_rows and len(Xtr) > max_train_rows:
        idx = np.random.default_rng(seed).choice(len(Xtr), max_train_rows, replace=False)
        Xtr, mtr = Xtr.iloc[idx], mtr.iloc[idx]
    cat_cols = [c for c, dt in Xtr.dtypes.items() if dt == object]
    Xtr_d = one_hot(Xtr, cat_cols)
    columns = list(Xtr_d.columns)
    Xte_d = one_hot(Xte, cat_cols, columns)
    ttr, ytr = mtr["t"].to_numpy(), mtr["y"].to_numpy()
    print(f"{log}: train {len(Xtr_d)} prefixes ({ttr.mean():.1%} treated), {len(columns)} one-hot features")

    prop = fit_propensity(Xtr_d, ttr, seed)
    w = ipw_weights(prop.predict_proba(Xtr_d.to_numpy(dtype=np.float32))[:, 1], ttr)
    arms = {}
    for arm, mask in (("treated", ttr == 1), ("untreated", ttr == 0)):
        if len(np.unique(ytr[mask])) < 2:
            arms[arm] = ConstantArm(float(ytr[mask].mean()))
            print(f"  {arm} arm: single outcome class in training ({ytr[mask].mean():.0%} deviant) -> constant arm")
            continue
        clf = XGBClassifier(**XGB_PARAMS)
        clf.fit(Xtr_d[mask], ytr[mask], sample_weight=w[mask])
        arms[arm] = clf
    pT, pU = arms["treated"].predict_proba(Xte_d)[:, 1], arms["untreated"].predict_proba(Xte_d)[:, 1]
    tte, yte = mte["t"].to_numpy(), mte["y"].to_numpy()

    manifest = {
        "log": log, "seed": seed, "n_train": int(len(Xtr_d)), "n_test": int(len(Xte_d)), "n_features": len(columns),
        "treated_share_train": float(ttr.mean()), "propensity_best": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in prop.get_params().items() if k in ("C", "penalty")},
        "auc_treated_arm_on_treated_test": float(roc_auc_score(yte[tte == 1], pT[tte == 1])) if len(set(yte[tte == 1])) > 1 else None,
        "auc_untreated_arm_on_untreated_test": float(roc_auc_score(yte[tte == 0], pU[tte == 0])) if len(set(yte[tte == 0])) > 1 else None,
        "mean_pT": float(pT.mean()), "mean_pU": float(pU.mean()), "share_cate_positive_rule": float(cp.positive_effect_rule(pT, pU).mean()),
        "settings": "CausalLift two-model estimator: XGBClassifier(max_depth=3, n_estimators=100, lr=0.1) per arm with IPW from a "
                    "LogisticRegression propensity (grid C in {0.1,1,10}, l1/l2), as in Shoush & Dumas causal/causallift_adapted.py",
        "elapsed_seconds": time.time() - t0, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if log == "SimBank":
        manifest["prefix_restriction"] = f"prefixes before the case's contact/skip decision and <= {SIMBANK_COMMON_PREFIX} (the events both arms share)"
    # agreement with the shipped counterfactual probabilities (BPIC only)
    rl = None if conf["rl_csv"] is None else pd.read_csv(conf["rl_csv"], sep=";", usecols=[conf["rl_case_col"], "prefix_nr", "Proba_if_Treated", "Proba_if_Untreated"])
    mm = mte.assign(pT=pT, pU=pU)
    if rl is not None:
        rl[conf["rl_case_col"]] = rl[conf["rl_case_col"]].astype(str)
    j = mm.iloc[:0] if rl is None else rl.merge(mm[["case_id", "prefix_nr", "pT", "pU"]], left_on=[conf["rl_case_col"], "prefix_nr"], right_on=["case_id", "prefix_nr"], how="inner")
    if len(j):
        manifest.update({
            "rl_csv_overlap_rows": int(len(j)),
            "corr_pT_with_shipped": float(np.corrcoef(j.pT, j.Proba_if_Treated)[0, 1]),
            "corr_pU_with_shipped": float(np.corrcoef(j.pU, j.Proba_if_Untreated)[0, 1]),
            "agreement_cate_sign_rule": float((cp.positive_effect_rule(j.pT, j.pU) == cp.positive_effect_rule(j.Proba_if_Treated, j.Proba_if_Untreated)).mean()),
        })
    mp = model_paths(log)
    mp["dir"].mkdir(parents=True, exist_ok=True)
    for arm in ("treated", "untreated"):
        for k in (arm, f"const_{arm}"):
            mp[k].unlink(missing_ok=True)
        if isinstance(arms[arm], ConstantArm):
            mp[f"const_{arm}"].write_text(json.dumps({"p": arms[arm].p}))
            manifest[f"{arm}_arm_constant"] = arms[arm].p
        else:
            arms[arm].save_model(str(mp[arm]))
    mp["propensity"].write_bytes(pickle.dumps(prop))
    mp["features"].write_text(json.dumps({"columns": columns, "cat_cols": cat_cols, "raw_columns": list(Xtr.columns),
                                          "families": rm.feature_families(list(Xtr.columns), conf)}, indent=2))
    mp["manifest"].write_text(json.dumps(manifest, indent=2, default=float))
    print(json.dumps(manifest, indent=1, default=float))
    return manifest


def load_model(log: str):
    from xgboost import XGBClassifier

    mp = model_paths(log)
    arms = {}
    for arm in ("treated", "untreated"):
        if mp[f"const_{arm}"].exists():
            arms[arm] = ConstantArm(json.loads(mp[f"const_{arm}"].read_text())["p"])
            continue
        clf = XGBClassifier()
        clf.load_model(str(mp[arm]))
        arms[arm] = clf
    feats = json.loads(mp["features"].read_text())
    return arms, feats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", default=list(EFFECT_LOGS), choices=list(EFFECT_LOGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-rows", type=int, default=None)
    a = ap.parse_args()
    for lg in a.logs:
        train(lg, a.seed, a.max_train_rows)
