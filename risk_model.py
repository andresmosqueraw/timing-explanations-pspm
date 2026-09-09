"""The outcome ("risk") predictor behind the policy's state, retrained.

The policy's ``reliability`` and ``deviation`` features are not raw log
attributes: Shoush & Dumas's offline phase (``run_offline_phase.sh`` ->
``predictive_model/get_catboost_pred_uncer.py`` -> ``prepare_data_for_RL_V2.py``)
trains a CatBoost outcome classifier on every prefix of the log (static +
aggregate encoding, label ``deviant``), and derives

    predicted   = argmax class of the classifier (ensemble size 1 in the
                  released script, so "ensemble mode" = the single model)
    deviation   = 1 - predicted
    reliability = ensemble agreement on ``deviation``   (BPIC2012: == deviation)
                  / probability of the predicted class   (BPIC2017 CSV)

This module retrains that classifier per log, from the same prepared logs
(``paths.PREPARED_LOGS``) with the same split (temporal 50/50, then a random
half of the test cases as validation, seed 22), the same encoding and the
same CatBoost settings, so that the risk score

    r(x_t) = P(deviant | prefix x_t)

can be *explained* -- the question the predictive-process-monitoring
explainability literature answers with SHAP over exactly this kind of model
-- and linked to the policy's timing decision on the same decision points.
For SimBank, whose ``reliability``/``deviation`` are the simulator's own
quality estimate and uncertainty, the same classifier is trained on the
log's observable event attributes with ``deviant`` = the case ends in
``cancel_application`` (negative profit), so all three logs get a risk model
of the same form.

Encoding. Shoush & Dumas materialise every prefix as its own trace
(``DatasetManager.generate_prefix_data``) and then aggregate; on BPIC2017
that is ~23M event rows. ``encode_prefixes`` computes the identical features
incrementally along each case (cumulative min/max/sum/mean/std of the numeric
attributes; the lexicographic running maximum of each categorical attribute,
which is what ``AggregateTransformer`` with ``boolean=True`` and ``model=
"catboost"`` computes -- ``groupby().max()`` on string columns; first-event
values of the static attributes). ``test_risk_model.py`` checks the two
agree on a materialised sample.

Usage:
    python risk_model.py                      # train all three, write models/risk/
    python risk_model.py --logs BPIC2012
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

import paths
import pools

sys.path.insert(0, str(paths.REPO / "foreign/common_files"))

# ---------------------------------------------------------------------------
# Per-log configuration (Shoush & Dumas's dataset_confs, plus SimBank)
# ---------------------------------------------------------------------------

LEAK_NOTE = """One deliberate departure from Shoush & Dumas's feature configuration:
their dynamic categorical columns include ``time_to_event_m``, the time
remaining until the case's treatment event, computed from the case's future.
With it the retrained BPIC2017 predictor reaches a test AUC of 0.94 and that
single column carries 71% of the model's importance, so a risk explanation
would mostly say "the treatment is N minutes away" -- information no process
actor has at the prefix. The risk models here exclude it (AUC drops
accordingly; see each manifest). Everything else follows their configuration.
"""

RISK_LOGS: dict[str, dict] = {
    "BPIC2012": {
        "shoush_name": "bpic2012",
        "case_col": "Case ID", "ts_col": "start_time", "activity_col": "Activity",
        "label_col": "label", "pos_label": "deviant",
        "dynamic_cat": ["Activity", "Resource"],  # released config also has time_to_event_m; see LEAK_NOTE
        "static_cat": ["event"],
        "dynamic_num": ["timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour", "open_cases"],
        "static_num": ["NumberOfOffers", "AMOUNT_REQ"],
        "rl_csv": paths.BPIC2012_CSV, "rl_case_col": "case_id",
    },
    "BPIC2017": {
        "shoush_name": "bpic2017",
        "case_col": "Case ID", "ts_col": "time:timestamp", "activity_col": "Activity",
        "label_col": "label", "pos_label": "deviant",
        "dynamic_cat": ["Activity", "org:resource", "Action", "EventOrigin", "lifecycle:transition", "Accepted", "Selected"],  # + time_to_event_m in the released config; see LEAK_NOTE
        "static_cat": ["ApplicationType", "LoanGoal"],
        "dynamic_num": ["FirstWithdrawalAmount", "MonthlyCost", "NumberOfTerms", "OfferedAmount", "CreditScore", "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour", "open_cases"],
        "static_num": ["NumberOfOffers", "RequestedAmount"],
        "rl_csv": paths.BPIC2017_CSV, "rl_case_col": "case_id",
    },
    "SimBank": {
        "shoush_name": None,
        "case_col": "case_nr", "ts_col": "synthetic_time_days", "activity_col": "activity",
        "label_col": "label", "pos_label": "deviant",
        # observable attributes only: the latent true `quality` and the
        # terminal `outcome` are excluded, est_/unc_quality are what the
        # bank sees (the policy's reliability/deviation are these two /10, /5)
        "dynamic_cat": ["activity"],
        "static_cat": [],
        "dynamic_num": ["est_quality", "unc_quality", "cum_cost", "interest_rate", "discount_factor", "noc", "nor", "elapsed_time", "event_nr"],
        "static_num": ["amount"],
        "rl_csv": None, "rl_case_col": "case_nr",
    },
}

# Stevens & De Smedt's three attribute families, on the encoded feature names
# (their filters: 'Activity' -> control flow, 'static' -> case, other 'agg' -> event).
def feature_families(feature_names: list[str], conf: dict) -> dict[str, list[str]]:
    static = set(conf["static_cat"]) | set(conf["static_num"])
    fam = {"event": [], "case": [], "control_flow": []}
    for f in feature_names:
        base = f.rsplit("_", 1)[0] if any(f.endswith(s) for s in ("_mean", "_max", "_min", "_sum", "_std")) else f
        if base == conf["activity_col"]:
            fam["control_flow"].append(f)
        elif base in static:
            fam["case"].append(f)
        else:
            fam["event"].append(f)
    return fam


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_events(log: str) -> tuple[pd.DataFrame, dict]:
    conf = RISK_LOGS[log]
    if conf["shoush_name"]:
        import dataset_confs  # vendored, points at paths.PREPARED_LOGS

        cat = conf["dynamic_cat"] + conf["static_cat"]
        dtypes = {c: "object" for c in cat + [conf["case_col"], conf["label_col"], conf["ts_col"]]}
        for c in conf["dynamic_num"] + conf["static_num"]:
            dtypes[c] = "float"
        df = pd.read_csv(dataset_confs.filename[conf["shoush_name"]], sep=";", dtype=dtypes, low_memory=False)
        df[conf["ts_col"]] = pd.to_datetime(df[conf["ts_col"]], format="mixed")
    else:
        df = pools.load_simbank(paths.SIMBANK_PKL).copy()
        last = df.groupby("case_nr")["activity"].transform("last")
        df["label"] = np.where(last == "cancel_application", "deviant", "regular")
        df["event_nr"] = df["prefix_nr"].astype(float)
        # case-level treatment flag and the prefix number of the case's
        # contact/skip decision (NaN for the priority cases that never reach
        # it), in Shoush & Dumas's vocabulary so effect_model.py reads both
        # logs the same way. In this log every contact happens at event 5 and
        # every skip at event 9 or 10 (verified), see effect_model.effect_rows.
        treated = df.groupby("case_nr")["activity"].transform(lambda a: (a == pools.SIMBANK_TREATMENT_ACTIVITY).any())
        df["treatment"] = np.where(treated, "treat", "noTreat")
        dec = df[df["activity"].isin(pools.DECISION_ACTIVITIES)].drop_duplicates("case_nr").set_index("case_nr")["prefix_nr"]
        df["decision_prefix"] = df["case_nr"].map(dec).astype(float)
        for c in conf["dynamic_num"] + conf["static_num"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        df["activity"] = df["activity"].astype(str)
    return df, conf


# ---------------------------------------------------------------------------
# Split (DatasetManager.split_data_strict + split_val, verbatim semantics)
# ---------------------------------------------------------------------------


def temporal_split(df: pd.DataFrame, conf: dict, train_ratio: float = 0.5, val_ratio: float = 0.5, seed: int = 22):
    case, ts, act = conf["case_col"], conf["ts_col"], conf["activity_col"]
    df = df.sort_values([ts, act], kind="mergesort")
    starts = df.groupby(case)[ts].min().reset_index().sort_values(ts, kind="mergesort")
    train_ids = list(starts[case])[: int(train_ratio * len(starts))]
    is_train = df[case].isin(train_ids)
    train, test = df[is_train], df[~is_train]
    split_ts = test[ts].min()
    train = train[train[ts] < split_ts]  # "strict": drop training events after the split
    # validation = random val_ratio of the test cases (split_val, split="random")
    tstarts = test.groupby(case)[ts].min().reset_index()
    np.random.seed(seed)
    tstarts = tstarts.reindex(np.random.permutation(tstarts.index))
    val_ids = list(tstarts[case])[-int(val_ratio * len(tstarts)):]
    is_val = test[case].isin(val_ids)
    return train, test[~is_val], test[is_val]


# ---------------------------------------------------------------------------
# Incremental static + aggregate encoding of every prefix
# ---------------------------------------------------------------------------


def encode_prefixes(df: pd.DataFrame, conf: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per (case, prefix length): the features Shoush & Dumas's
    static + agg encoders produce for that prefix. Returns (X, meta) where
    meta has case id, prefix_nr, case_length and the numeric label."""
    case, ts, act = conf["case_col"], conf["ts_col"], conf["activity_col"]
    df = df.sort_values([case, ts, act], kind="mergesort").reset_index(drop=True)
    g = df.groupby(case, sort=False)
    out = {}
    # static: first event's values (StaticTransformer)
    for c in conf["static_num"]:
        out[c] = g[c].transform("first").fillna(0.0).astype(float)
    for c in conf["static_cat"]:
        out[c] = g[c].transform("first").fillna("0").astype(str)
    # aggregate categoricals: running lexicographic max (AggregateTransformer,
    # boolean=True, catboost branch: groupby().max() on the raw strings)
    for c in conf["dynamic_cat"]:
        s = df[c].fillna("0").astype(str)
        cats = pd.Categorical(s, categories=sorted(s.unique()), ordered=True)
        codes = pd.Series(cats.codes, index=df.index)
        run = codes.groupby(df[case], sort=False).cummax()
        out[c] = pd.Series(np.asarray(cats.categories)[run.to_numpy()], index=df.index)
    # aggregate numerics: mean, max, min, sum, std over the prefix
    for c in conf["dynamic_num"]:
        s = df[c].astype(float)
        gs = s.groupby(df[case], sort=False)
        n = gs.cumcount() + 1
        csum = gs.cumsum()
        csum2 = (s ** 2).groupby(df[case], sort=False).cumsum()
        mean = csum / n
        var = (csum2 - n * mean ** 2) / (n - 1).replace(0, np.nan)
        std = np.sqrt(var.clip(lower=0)).fillna(0.0)  # pandas agg std: ddof=1, NaN for n=1 -> filled with 0
        out[f"{c}_mean"] = mean.fillna(0.0)
        out[f"{c}_max"] = gs.cummax().fillna(0.0)
        out[f"{c}_min"] = gs.cummin().fillna(0.0)
        out[f"{c}_sum"] = csum.fillna(0.0)
        out[f"{c}_std"] = std
    X = pd.DataFrame(out, index=df.index)
    meta = pd.DataFrame({
        "case_id": df[case].astype(str),
        "prefix_nr": g.cumcount() + 1,
        "case_length": g[case].transform("size"),
        "y": (df[conf["label_col"]] == conf["pos_label"]).astype(int),
        "timestamp": df[ts],
    }, index=df.index)
    # the case's treatment flag (Shoush & Dumas: "treat"/"noTreat"; on SimBank
    # built in load_events), used by effect_model.py's two-model estimator
    if "treatment" in df.columns:
        meta["t"] = (df["treatment"].astype(str) == "treat").astype(int)
    if "decision_prefix" in df.columns:
        meta["decision_prefix"] = df["decision_prefix"].astype(float)
    return X, meta


def cat_feature_indices(X: pd.DataFrame) -> list[int]:
    return [i for i, dt in enumerate(X.dtypes) if dt != float]


# ---------------------------------------------------------------------------
# CatBoost (get_catboost_pred_uncer.py's Ensemble(esize=1) member, verbatim)
# ---------------------------------------------------------------------------


def make_classifier(seed: int = 2, iterations: int = 1000):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=iterations, depth=6, border_count=128, random_strength=100,
        loss_function="Logloss", verbose=False, bootstrap_type="Bernoulli",
        posterior_sampling=True, eval_metric="AUC", use_best_model=True,
        langevin=True, random_seed=seed, thread_count=-1,
    )


def model_paths(log: str) -> dict:
    d = paths.RISK_MODELS
    return {"model": d / f"{log}_catboost.cbm", "manifest": d / f"{log}_manifest.json", "features": d / f"{log}_features.json"}


def train(log: str, iterations: int = 1000, seed: int = 2, max_train_rows: int | None = None) -> dict:
    from sklearn.metrics import roc_auc_score

    t0 = time.time()
    df, conf = load_events(log)
    tr, te, va = temporal_split(df, conf)
    Xtr, mtr = encode_prefixes(tr, conf)
    Xva, mva = encode_prefixes(va, conf)
    Xte, mte = encode_prefixes(te, conf)
    if max_train_rows and len(Xtr) > max_train_rows:
        idx = np.random.default_rng(seed).choice(len(Xtr), max_train_rows, replace=False)
        Xtr, mtr = Xtr.iloc[idx], mtr.iloc[idx]
    cat_idx = cat_feature_indices(Xtr)
    print(f"{log}: train {len(Xtr)} / val {len(Xva)} / test {len(Xte)} prefixes, {Xtr.shape[1]} features "
          f"({len(cat_idx)} categorical), deviant share train={mtr.y.mean():.3f}")
    clf = make_classifier(seed, iterations)
    clf.fit(Xtr, mtr.y, cat_features=cat_idx, eval_set=(Xva, mva.y))
    p_te = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(mte.y, p_te))
    manifest = {
        "log": log, "seed": seed, "iterations": iterations, "best_iteration": int(clf.get_best_iteration()),
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)), "n_features": int(Xtr.shape[1]),
        "deviant_share_train": float(mtr.y.mean()), "test_auc": auc,
        "feature_importance_top10": dict(sorted(zip(Xtr.columns, clf.get_feature_importance()), key=lambda kv: -kv[1])[:10]),
        "elapsed_seconds": time.time() - t0, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": "CatBoostClassifier(iterations, depth=6, border_count=128, random_strength=100, Logloss, Bernoulli, "
                    "posterior_sampling, eval AUC, use_best_model, langevin) = Shoush & Dumas get_catboost_pred_uncer.py, ensemble size 1",
        "label": conf["pos_label"] if conf["shoush_name"] else "deviant = case ends in cancel_application (SimBank, ours)",
    }
    # agreement with the probabilities Shoush & Dumas shipped in the RL CSV
    if conf["rl_csv"] is not None:
        rl = pd.read_csv(conf["rl_csv"], sep=";", usecols=[conf["rl_case_col"], "prefix_nr", "predicted_proba_1", "predicted", "actual"])
        rl[conf["rl_case_col"]] = rl[conf["rl_case_col"]].astype(str)
        allX = pd.concat([Xte, Xva]); allm = pd.concat([mte, mva])
        allm = allm.assign(p=clf.predict_proba(allX)[:, 1])
        j = rl.merge(allm[["case_id", "prefix_nr", "p", "y"]], left_on=[conf["rl_case_col"], "prefix_nr"], right_on=["case_id", "prefix_nr"], how="inner")
        if len(j):
            manifest["rl_csv_overlap_rows"] = int(len(j))
            manifest["corr_with_shipped_proba"] = float(np.corrcoef(j.p, j.predicted_proba_1)[0, 1])
            manifest["agreement_with_shipped_predicted"] = float(((j.p > 0.5).astype(int) == j.predicted).mean())
            manifest["label_matches_actual"] = float((j.y == j.actual).mean())
            manifest["shipped_auc_on_overlap"] = float(roc_auc_score(j.actual, j.predicted_proba_1)) if j.actual.nunique() > 1 else None
            manifest["retrained_auc_on_overlap"] = float(roc_auc_score(j.actual, j.p)) if j.actual.nunique() > 1 else None
    mp = model_paths(log)
    paths.RISK_MODELS.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(mp["model"]))
    mp["manifest"].write_text(json.dumps(manifest, indent=2, default=float))
    mp["features"].write_text(json.dumps({"feature_names": list(Xtr.columns), "cat_feature_indices": cat_idx,
                                          "families": feature_families(list(Xtr.columns), conf)}, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "feature_importance_top10"}, indent=1, default=float))
    return manifest


def load_model(log: str):
    from catboost import CatBoostClassifier

    mp = model_paths(log)
    clf = CatBoostClassifier()
    clf.load_model(str(mp["model"]))
    feats = json.loads(mp["features"].read_text())
    return clf, feats


def prefixes_for_pool(log: str, rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The encoded prefixes of the policy's evaluation-pool rows (matched by
    case id and prefix number), so risk and timing are read on the same
    decision points. Returns (X, meta) aligned with ``rows``' order; rows
    whose case is missing from the prepared log are dropped."""
    df, conf = load_events(log)
    X, meta = encode_prefixes(df, conf)
    key_col = conf["rl_case_col"]
    want = pd.DataFrame({"case_id": rows[key_col].astype(str).to_numpy(), "prefix_nr": rows["prefix_nr"].astype(int).to_numpy(), "_order": np.arange(len(rows))})
    meta = meta.assign(_i=np.arange(len(meta)))
    j = want.merge(meta[["case_id", "prefix_nr", "_i"]], on=["case_id", "prefix_nr"], how="inner").sort_values("_order")
    return X.iloc[j["_i"].to_numpy()].reset_index(drop=True), meta.iloc[j["_i"].to_numpy()].reset_index(drop=True).assign(pool_row=j["_order"].to_numpy())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", default=list(RISK_LOGS), choices=list(RISK_LOGS))
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--max-train-rows", type=int, default=None, help="subsample the training prefixes (smoke runs)")
    a = ap.parse_args()
    for lg in a.logs:
        train(lg, a.iterations, a.seed, a.max_train_rows)
