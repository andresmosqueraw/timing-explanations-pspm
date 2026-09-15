"""Scores of the risk and effect models on each split, without leakage.

The agent must train on the training split only, but the risk and effect
models were fit on that same split: their in-sample scores there are more
confident than anything the agent will read on new cases. So the training
split is scored by *cross-fitting*: its cases are split into ``N_FOLDS``
folds, and each fold is scored by a risk model and an effect estimator fit
on the other folds only (out-of-fold predictions). Every training prefix
thus gets scores from models that never saw its case.

The fold models use the settings of the final models (risk_model.make_classifier
with the same seed and the validation split for early stopping; the effect
estimator with the propensity's C and penalty chosen on the full training
split, fit without the grid). The validation and test splits are scored by
the final models, fit on the whole training split.

    split  scored by                         used for
    train  out-of-fold models (N_FOLDS)       training the agent
    val    final models                      early stopping, choosing the agent
    test   final models                      final gain and every explanation
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import effect_model as em
import risk_model as rm

N_FOLDS = 5
FOLD_SEED = 0


def case_folds(case_ids: pd.Series, n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> np.ndarray:
    """Fold index per row; every case falls in exactly one fold."""
    cases = pd.Series(case_ids.astype(str).unique())
    perm = np.random.default_rng(seed).permutation(len(cases))
    fold_of = dict(zip(cases.iloc[perm], np.arange(len(cases)) % n_folds))
    return case_ids.astype(str).map(fold_of).to_numpy()


def _final_scores(log: str, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clf, rfeats = rm.load_model(log)
    arms, efeats = em.load_model(log)
    r = clf.predict_proba(X[rfeats["feature_names"]])[:, 1]
    Xd = em.one_hot(X[efeats["raw_columns"]], efeats["cat_cols"], efeats["columns"])
    return r, arms["treated"].predict_proba(Xd)[:, 1], arms["untreated"].predict_proba(Xd)[:, 1]


def _fit_effect(log: str, Xd: pd.DataFrame, meta: pd.DataFrame, prop_params: dict, seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    keep = em.effect_rows(log, meta)
    Xd, meta = Xd[keep], meta[keep]
    t, y = meta["t"].to_numpy(), meta["y"].to_numpy()
    prop = LogisticRegression(solver="liblinear", max_iter=1000, random_state=seed, **prop_params)
    Xa = np.ascontiguousarray(Xd.to_numpy(dtype=np.float32))
    prop.fit(Xa, t)
    w = em.ipw_weights(prop.predict_proba(Xa)[:, 1], t)
    arms = {}
    for arm, mask in (("treated", t == 1), ("untreated", t == 0)):
        if len(np.unique(y[mask])) < 2:
            arms[arm] = em.ConstantArm(float(y[mask].mean()))
            continue
        clf = XGBClassifier(**em.XGB_PARAMS)
        clf.fit(Xd[mask], y[mask], sample_weight=w[mask])
        arms[arm] = clf
    return arms


def score_split(log: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """(X, meta, r, p_T, p_U, fold, training split, info) for ``split``.
    ``fold`` is -1 outside the training split."""
    df, conf = rm.load_events(log)
    tr, te, va = rm.temporal_split(df, conf)
    part = {"train": tr, "val": va, "test": te}[split]
    X, meta = rm.encode_prefixes(part, conf)
    if split != "train":
        r, pT, pU = _final_scores(log, X)
        return X, meta, r, pT, pU, np.full(len(X), -1), tr, {}

    rfeats = json.loads(rm.model_paths(log)["features"].read_text())
    efeats = json.loads(em.model_paths(log)["features"].read_text())
    emani = json.loads(em.model_paths(log)["manifest"].read_text())
    rmani = json.loads(rm.model_paths(log)["manifest"].read_text())
    prop_params = {k: emani["propensity_best"][k] for k in ("C", "penalty")}
    Xva, mva = rm.encode_prefixes(va, conf)
    fold = case_folds(meta["case_id"])
    r, pT, pU = np.full(len(X), np.nan), np.full(len(X), np.nan), np.full(len(X), np.nan)
    Xr = X[rfeats["feature_names"]]
    Xd = em.one_hot(X[efeats["raw_columns"]], efeats["cat_cols"], efeats["columns"])
    info = {"n_folds": N_FOLDS, "fold_seed": FOLD_SEED, "propensity_params": prop_params, "risk_seed": rmani["seed"],
            "risk_iterations": rmani["iterations"], "folds": []}
    for k in range(N_FOLDS):
        t0 = time.time()
        fit, held = fold != k, fold == k
        clf = rm.make_classifier(rmani["seed"], rmani["iterations"])
        clf.fit(Xr[fit], meta["y"][fit], cat_features=rfeats["cat_feature_indices"],
                eval_set=(Xva[rfeats["feature_names"]], mva["y"]))
        r[held] = clf.predict_proba(Xr[held])[:, 1]
        arms = _fit_effect(log, Xd[fit], meta[fit], prop_params)
        pT[held] = arms["treated"].predict_proba(Xd[held])[:, 1]
        pU[held] = arms["untreated"].predict_proba(Xd[held])[:, 1]
        info["folds"].append({"fold": k, "n_fit": int(fit.sum()), "n_scored": int(held.sum()), "risk_best_iteration": int(clf.get_best_iteration()),
                              "cases_scored": int(meta["case_id"][held].nunique()), "seconds": time.time() - t0})
        print(f"  {log} fold {k}: fit {fit.sum()} / scored {held.sum()} prefixes in {time.time() - t0:.0f}s", flush=True)
    assert not (np.isnan(r).any() or np.isnan(pT).any() or np.isnan(pU).any())
    return X, meta, r, pT, pU, fold, tr, info
