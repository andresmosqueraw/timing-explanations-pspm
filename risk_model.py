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

LEAK_NOTE = """Departures from Shoush & Dumas's configuration, all to keep the future
of a case out of its prefix (``leakage_audit.py`` checks every one):

* ``time_to_event_m``: the time remaining until the case's outcome event.
* ``NumberOfOffers`` (BPIC2012/2017): the number of offers of the *whole*
  case, copied onto every event; it reveals offers not yet made on 35-53%
  of the test prefixes, and it is the very column the released treatment is
  a function of (``treat`` iff the case gets at most one offer).
* ``CreditScore`` (BPIC2017): non-zero only on the offer the applicant ends
  up accepting (P(deviant | CreditScore != 0 seen) = 0.00006 on 474k
  prefixes), so it announces the outcome at offer creation. ``Accepted`` and
  ``Selected`` are attributes of the same offer object whose recording time
  is not documented; they are dropped with it. The offer's terms
  (OfferedAmount, MonthlyCost, NumberOfTerms, FirstWithdrawalAmount) are set
  when the offer is created and carry no such signal, and are kept.
* Outcome-revealing prefixes: a case's decision points end before its first
  ``outcome_activities`` event (e.g. O_Accepted: P(deviant) = 0 afterwards),
  and the models are trained on those prefixes only.
* Treatment: ``treatment_activity`` occurring for the ``treatment_nth`` time
  before the outcome (BPIC: a further offer, the 2nd one; Sepsis: IV
  Antibiotics) makes the case treated (T = 1), and its decision points end
  before that event, so every covariate is pre-treatment. The released
  treatment column (BPIC: "treat" = at most one offer) is replaced.
* Static attributes take the value known so far in the prefix, never a value
  first recorded at a later event.
"""

RISK_LOGS: dict[str, dict] = {
    "BPIC2012": {
        "shoush_name": "bpic2012",
        "case_col": "Case ID", "ts_col": "start_time", "activity_col": "Activity",
        "label_col": "label", "pos_label": "deviant",
        "dynamic_cat": ["Activity", "Resource"],  # released config also has time_to_event_m; see LEAK_NOTE
        "static_cat": ["event"],
        "dynamic_num": ["timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour", "open_cases"],
        "static_num": ["AMOUNT_REQ"],  # released config also has NumberOfOffers; see LEAK_NOTE
        # decision points end before the first event that reveals the outcome ...
        "outcome_activities": ["A_APPROVED", "A_REGISTERED", "A_ACTIVATED", "A_CANCELLED", "A_DECLINED", "O_ACCEPTED"],
        # ... and before the intervention: a further offer, the case's 2nd O_SENT
        "treatment_activity": "O_SENT", "treatment_nth": 2,
        "rl_csv": paths.BPIC2012_CSV, "rl_case_col": "case_id",
    },
    "BPIC2017": {
        "shoush_name": "bpic2017",
        "case_col": "Case ID", "ts_col": "time:timestamp", "activity_col": "Activity",
        "label_col": "label", "pos_label": "deviant",
        # released config also has time_to_event_m, Accepted, Selected, CreditScore and NumberOfOffers; see LEAK_NOTE
        "dynamic_cat": ["Activity", "org:resource", "Action", "EventOrigin", "lifecycle:transition"],
        "static_cat": ["ApplicationType", "LoanGoal"],
        "dynamic_num": ["FirstWithdrawalAmount", "MonthlyCost", "NumberOfTerms", "OfferedAmount", "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour", "open_cases"],
        "static_num": ["RequestedAmount"],
        "outcome_activities": ["A_Pending", "A_Denied", "A_Cancelled", "O_Accepted"],
        "treatment_activity": "O_Created", "treatment_nth": 2,
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
    "Sepsis": {
        # "Sepsis Cases - Event Log" (Mannhardt, 4TU.ResearchData). Outcome
        # ("risk") = the case revisits the ER (activity "Return ER" occurs
        # anywhere in the trace; verified to fall at the case's last event in
        # 99.9% of the 294 occurrences, so it is a genuine case outcome, not a
        # mid-case waypoint). Intervention = "IV Antibiotics": the case is
        # treated if it receives them before returning, and its decision
        # points are the prefixes before that (add_decision_points), as on BPIC.
        "shoush_name": None,
        "case_col": "case_id", "ts_col": "timestamp", "activity_col": "activity",
        "label_col": "label", "pos_label": "deviant",
        "dynamic_cat": ["activity", "org:group"],
        # a handful of the SIRS/diagnostic flags recorded once at ER
        # Registration (case-level; encode_prefixes' "first non-null per
        # case" static rule picks them up correctly even though every later
        # event carries None for these columns, see load_events)
        "static_cat": ["InfectionSuspected", "DiagnosticBlood", "SIRSCriteria2OrMore",
                       "DisfuncOrg", "Hypotensie", "Infusion", "Oligurie", "Hypoxie"],
        "dynamic_num": ["Leucocytes", "CRP", "LacticAcid", "timesincelastevent", "timesincecasestart", "hour", "weekday", "event_nr"],
        "static_num": ["Age"],
        "outcome_activities": ["Return ER"],
        "treatment_activity": "IV Antibiotics", "treatment_nth": 1,
        "rl_csv": None, "rl_case_col": "case_id",
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
# Sepsis Cases - Event Log
# ---------------------------------------------------------------------------


def load_sepsis_events() -> pd.DataFrame:
    """The parsed Sepsis log (``paths.SEPSIS_EVENTS_PARQUET``) with the
    derived columns ``encode_prefixes`` needs: the outcome label, the
    *dynamic* per-prefix treatment flag, and timestamp-derived dynamic
    numerics (Shoush & Dumas's own dynamic_num vocabulary has no equivalent
    of a raw event timestamp column to reuse, so these are computed here,
    the same way ``dataset_confs``/``DatasetManager`` compute them upstream
    of the CSVs risk_model reads for BPIC)."""
    conf = RISK_LOGS["Sepsis"]
    df = pd.read_parquet(paths.SEPSIS_EVENTS_PARQUET)
    df = df.sort_values(["case_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("case_id", sort=False)
    df["event_nr"] = (g.cumcount() + 1).astype(float)

    # label: the case ever revisits the ER. "Return ER" falls at the case's
    # last event in 99.9% of its 294 occurrences (verified on this log), so
    # this is a case outcome, not a mid-trace waypoint whose later prefixes
    # would otherwise trivially "know" the label before it is reached.
    ever_return = g["activity"].transform(lambda a: (a == "Return ER").any())
    df["label"] = np.where(ever_return, "deviant", "regular")

    # treatment and decision points: add_decision_points (load_events)

    ts = df["timestamp"]
    case_start = g["timestamp"].transform("min")
    prev_ts = g["timestamp"].shift(1)
    df["timesincecasestart"] = (ts - case_start).dt.total_seconds() / 3600.0
    df["timesincelastevent"] = (ts - prev_ts).dt.total_seconds().fillna(0.0) / 60.0
    df["hour"] = ts.dt.hour.astype(float)
    df["weekday"] = ts.dt.weekday.astype(float)

    for c in conf["dynamic_num"] + conf["static_num"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["activity"] = df["activity"].astype(str)
    df["org:group"] = df["org:group"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def add_decision_points(df: pd.DataFrame, conf: dict) -> pd.DataFrame:
    """Leak-free treatment flag and decision-point mask, when the log config
    names its outcome and treatment activities (see LEAK_NOTE).

    In the event order ``encode_prefixes`` uses, an event is a decision point
    iff it comes strictly before the case's first outcome-revealing event and
    before the ``treatment_nth`` occurrence of the treatment activity; the
    case is treated iff that occurrence exists and precedes the outcome."""
    if "outcome_activities" not in conf:
        return df
    case, ts, act = conf["case_col"], conf["ts_col"], conf["activity_col"]
    order = df.sort_values([case, ts, act], kind="mergesort").index
    d = df.loc[order]
    pos = d.groupby(case, sort=False).cumcount().to_numpy()
    cases = d[case].to_numpy()
    is_out = d[act].isin(conf["outcome_activities"]).to_numpy()
    is_tr = (d[act] == conf["treatment_activity"]).to_numpy()
    tr_count = pd.Series(is_tr.astype(int), index=d.index).groupby(cases, sort=False).cumsum().to_numpy()
    big = np.iinfo(np.int64).max
    first_out = pd.Series(np.where(is_out, pos, big), index=d.index).groupby(cases, sort=False).transform("min").to_numpy()
    nth_tr = pd.Series(np.where(is_tr & (tr_count == conf["treatment_nth"]), pos, big), index=d.index).groupby(cases, sort=False).transform("min").to_numpy()
    treated = nth_tr < first_out
    df = df.copy()
    df.loc[order, "treatment"] = np.where(treated, "treat", "noTreat")
    df.loc[order, "_decision"] = pos < np.minimum(first_out, nth_tr)
    df["_decision"] = df["_decision"].astype(bool)
    return df


def progress_horizon(train: pd.DataFrame, conf: dict, q: float = 0.95) -> float:
    """The fixed denominator of the policy's relative_position: the q-quantile
    of decision points per case in the training split, so progress needs no
    knowledge of how long the current case will last."""
    dec = train["_decision"] if "_decision" in train.columns else pd.Series(True, index=train.index)
    return float(max(dec.groupby(train[conf["case_col"]]).sum().quantile(q), 1.0))


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
    elif log == "Sepsis":
        df = load_sepsis_events()
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
    return add_decision_points(df, conf), conf


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
    # static: the value known so far in the prefix (StaticTransformer takes the
    # case's first non-null value, which can come from a later event)
    for c in conf["static_num"]:
        out[c] = g[c].ffill().fillna(0.0).astype(float)
    for c in conf["static_cat"]:
        out[c] = g[c].ffill().fillna("0").astype(str)
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
    if "_decision" in df.columns:  # decision points only (add_decision_points)
        keep = df["_decision"].to_numpy(bool)
        X, meta = X[keep].reset_index(drop=True), meta[keep].reset_index(drop=True)
        meta["case_length"] = meta.groupby("case_id", sort=False)["prefix_nr"].transform("max")  # decision points per case
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
        "label": conf["pos_label"] if conf["shoush_name"] else (
            "deviant = case ends in cancel_application (SimBank, ours)" if log == "SimBank"
            else "deviant = case revisits the ER, activity 'Return ER' (Sepsis, ours)"),
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
