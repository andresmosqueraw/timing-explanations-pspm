"""risk_model.encode_prefixes must reproduce Shoush & Dumas's materialised
prefix encoding (DatasetManager.generate_prefix_data + StaticTransformer +
AggregateTransformer) feature for feature.

    pytest test_risk_model.py -v
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

import paths
import risk_model as rm

sys.path.insert(0, str(paths.REPO / "foreign/common_files"))


def _toy_log(seed=0, n_cases=12):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cases):
        L = int(rng.integers(1, 7))
        goal = rng.choice(["car", "home"])
        for k in range(L):
            rows.append({
                "Case ID": f"c{c}", "time:timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(hours=c * 100 + k),
                "Activity": rng.choice(["A_Create", "W_Call", "O_Offer"]), "Action": rng.choice(["x", "y"]),
                "LoanGoal": goal, "RequestedAmount": float(1000 * (c + 1)),
                "timesincecasestart": float(k * 1.5), "open_cases": float(rng.integers(0, 50)),
                "label": "deviant" if c % 3 == 0 else "regular",
            })
    return pd.DataFrame(rows)


CONF = {
    "shoush_name": None, "case_col": "Case ID", "ts_col": "time:timestamp", "activity_col": "Activity",
    "label_col": "label", "pos_label": "deviant",
    "dynamic_cat": ["Activity", "Action"], "static_cat": ["LoanGoal"],
    "dynamic_num": ["timesincecasestart", "open_cases"], "static_num": ["RequestedAmount"],
    "rl_csv": None, "rl_case_col": "Case ID",
}


def _materialised(df, tmp_path):
    """Shoush & Dumas's own pipeline on the same log."""
    from transformers.AggregateTransformer import AggregateTransformer
    from transformers.StaticTransformer import StaticTransformer

    case = CONF["case_col"]
    data = df.sort_values([CONF["ts_col"], CONF["activity_col"]], kind="mergesort").copy()
    data["case_length"] = data.groupby(case)[CONF["activity_col"]].transform(len)
    max_len = int(data["case_length"].max())
    prefixes = data.groupby(case).head(1).assign(prefix_nr=1, orig_case_id=lambda d: d[case])
    for n in range(2, max_len + 1):
        tmp = data[data["case_length"] >= n].groupby(case).head(n).copy()
        tmp["orig_case_id"] = tmp[case]
        tmp[case] = tmp[case] + f"_{n}"
        tmp["prefix_nr"] = n
        prefixes = pd.concat([prefixes, tmp])
    st = StaticTransformer(str(tmp_path), "toy", case_id_col=case, cat_cols=CONF["static_cat"], num_cols=CONF["static_num"])
    ag = AggregateTransformer(str(tmp_path), "toy", case_id_col=case, cat_cols=CONF["dynamic_cat"], num_cols=CONF["dynamic_num"])
    X = pd.concat([st.transform(prefixes).reset_index(drop=True), ag.transform(prefixes).reset_index(drop=True)], axis=1)
    keys = prefixes.groupby(case).first()[["orig_case_id", "prefix_nr"]].reset_index(drop=True)
    return X, keys


def test_incremental_encoding_matches_materialised(tmp_path):
    df = _toy_log()
    X_inc, meta = rm.encode_prefixes(df, CONF)
    X_mat, keys = _materialised(df, tmp_path)
    assert len(X_inc) == len(X_mat) == len(df)
    inc = X_inc.assign(case_id=meta.case_id.to_numpy(), prefix_nr=meta.prefix_nr.to_numpy()).set_index(["case_id", "prefix_nr"]).sort_index()
    mat = X_mat.assign(case_id=keys.orig_case_id.astype(str).to_numpy(), prefix_nr=keys.prefix_nr.to_numpy()).set_index(["case_id", "prefix_nr"]).sort_index()
    assert set(inc.columns) == set(mat.columns)
    for col in inc.columns:
        a, b = inc[col].to_numpy(), mat[col].to_numpy()
        if inc[col].dtype == object:
            assert (a.astype(str) == b.astype(str)).all(), col
        else:
            np.testing.assert_allclose(a.astype(float), b.astype(float), atol=1e-9, err_msg=col)


def test_label_and_meta():
    df = _toy_log()
    X, meta = rm.encode_prefixes(df, CONF)
    assert (meta.groupby("case_id").prefix_nr.max() == meta.groupby("case_id").case_length.first()).all()
    assert set(meta.y.unique()) <= {0, 1}
    assert rm.cat_feature_indices(X) == [i for i, c in enumerate(X.columns) if c in ("LoanGoal", "Activity", "Action")]


def test_families():
    names = ["RequestedAmount", "LoanGoal", "Activity", "Action", "timesincecasestart_mean", "open_cases_std"]
    fam = rm.feature_families(names, CONF)
    assert fam["control_flow"] == ["Activity"]
    assert fam["case"] == ["RequestedAmount", "LoanGoal"]
    assert fam["event"] == ["Action", "timesincecasestart_mean", "open_cases_std"]


def test_temporal_split_is_strict_and_disjoint():
    df = _toy_log(n_cases=20)
    tr, te, va = rm.temporal_split(df, CONF)
    c = CONF["case_col"]
    assert not (set(tr[c]) & set(te[c])) and not (set(tr[c]) & set(va[c])) and not (set(te[c]) & set(va[c]))
    assert tr[CONF["ts_col"]].max() < pd.concat([te, va])[CONF["ts_col"]].min()
