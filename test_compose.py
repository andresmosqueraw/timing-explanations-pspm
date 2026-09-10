"""Unit tests of the composition operators (compose.py), on toy inputs.

    pytest test_compose.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import compose as cp


def _toy_timing(n=7, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 6))


def _toy_lower(n=7, D=5, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "reliability": rng.normal(size=(n, D)) + 0.5,
        "deviation": None,  # filled below: same model as reliability
        "Proba_if_Treated": rng.normal(size=(n, D)) - 0.3,
        "Proba_if_Untreated": rng.normal(size=(n, D)) + 1.0,
    }


def test_risk_features_follow_the_offline_phase():
    r = np.array([0.1, 0.5, 0.7, 0.99])
    rel, dev = cp.risk_features_from_r("BPIC2017", r)
    assert dev.tolist() == [1.0, 1.0, 0.0, 0.0]  # deviation = 1 - 1[r > 0.5]
    assert np.allclose(rel, [0.9, 0.5, 0.7, 0.99])  # probability of the predicted class
    rel12, dev12 = cp.risk_features_from_r("BPIC2012", r)
    assert np.array_equal(rel12, dev12)  # ensemble agreement of one model


def test_rebuild_state_keeps_natives_and_replaces_the_rest():
    states = np.array([[0.2, 0.9, 1.0, 3.0, 0.1, 0.2], [0.8, 0.6, 0.0, 1.0, 0.7, 0.3]], np.float32)
    out = cp.rebuild_state("BPIC2017", states, r=np.array([0.9, 0.2]), pT=np.array([0.5, 0.6]), pU=np.array([0.4, 0.1]))
    assert np.allclose(out[:, [0, 3]], states[:, [0, 3]])
    assert out[0, 2] == 0.0 and out[1, 2] == 1.0
    assert np.allclose(out[:, 4], [0.5, 0.6]) and np.allclose(out[:, 5], [0.4, 0.1])


def test_propagation_conserves_completeness_and_channels():
    phi_t = _toy_timing()
    lower = _toy_lower()
    lower["deviation"] = lower["reliability"]
    out = cp.propagate(phi_t, lower)
    assert cp.completeness_gap(phi_t, out["composed"]) < 1e-9
    # each channel carries exactly the timing weight of its coordinates
    assert np.allclose(out["channels"]["risk"].sum(axis=1), phi_t[:, 1] + phi_t[:, 2])
    assert np.allclose(out["channels"]["effect_T"].sum(axis=1), phi_t[:, 4])
    assert np.allclose(out["channels"]["effect_U"].sum(axis=1), phi_t[:, 5])
    assert np.allclose(out["native"], phi_t[:, [0, 3]])
    assert out["composed"].shape == (7, 5 + 2)


def test_propagation_through_identity_lower_level_is_identity():
    phi_t = _toy_timing()
    D = 4
    ident = np.zeros((7, D)); ident[:, 2] = 1.0  # the lower model *is* attribute 2
    lower = {"reliability": ident, "deviation": ident, "Proba_if_Treated": ident, "Proba_if_Untreated": ident}
    out = cp.propagate(phi_t, lower)
    assert np.allclose(out["composed"][:, 2], phi_t[:, [1, 2, 4, 5]].sum(axis=1))
    assert np.allclose(out["composed"][:, [0, 1, 3]], 0.0)


def test_fallback_split_when_lower_sum_vanishes():
    phi_t = np.ones((1, 6))
    cancel = np.array([[1.0, -1.0, 0.0]])  # sums to zero: cannot divide
    lower = {"reliability": cancel, "deviation": cancel, "Proba_if_Treated": cancel, "Proba_if_Untreated": cancel}
    out = cp.propagate(phi_t, lower)
    assert out["fallbacks"] == {"reliability": 1, "deviation": 1, "Proba_if_Treated": 1, "Proba_if_Untreated": 1}
    assert np.allclose(out["composed"][0, :3], [2.0, 2.0, 0.0])  # 4 units split by |phi| over two attributes
    assert cp.completeness_gap(phi_t, out["composed"]) < 1e-9


def test_level_shares_sum_to_one():
    s = cp.level_shares(_toy_timing())
    assert abs(sum(s["per_feature"].values()) - 1) < 1e-9
    assert abs(sum(s["per_level"].values()) - 1) < 1e-9
    assert set(s["per_level"]) == {"native", "risk", "effect"}


def test_typology_cells_partition_the_pool():
    rng = np.random.default_rng(3)
    n = 40
    cells = cp.typology(rng.random(n) > 0.5, rng.random(n) > 0.5, rng.random(n) > 0.5, _toy_timing(n))
    assert len(cells) == 8 and sum(c["n"] for c in cells.values()) == n
    for c in cells.values():
        if c["n"]:
            assert abs(sum(c["level_share"].values()) - 1) < 1e-9


class _LinearBox:
    """A stand-in for EndToEndBox with f linear in two numeric columns."""

    columns = ["a", "b", "c"]
    cat_cols = {"c"}

    def f(self, Z, sign=1.0):
        return np.asarray(sign, float) * (2.0 * Z["a"].to_numpy(float) - Z["b"].to_numpy(float) + (Z["c"] == "x").to_numpy(float))

    def mask(self, Z, idx, ref):
        return cp.EndToEndBox.mask(self, Z, idx, ref)


def test_shapley_sampling_is_exact_for_a_linear_box():
    box = _LinearBox()
    Z = pd.DataFrame({"a": [1.0, 3.0], "b": [2.0, 0.0], "c": ["x", "y"]})
    ref = pd.Series({"a": 0.0, "b": 1.0, "c": "y"})
    sign = np.array([1.0, -1.0])
    phi = cp.shapley_sampling(box, Z, ref, sign, n_perm=3, seed=0)
    expect = np.array([[2.0, -1.0, 1.0], [-6.0, -1.0, 0.0]])  # sign * (2 da, -db, [c == x] - [ref == x])
    assert np.allclose(phi, expect)
    assert np.allclose(phi.sum(axis=1), box.f(Z, sign) - box.f(pd.DataFrame([ref, ref]), sign))


def test_deletion_test_e2e_guided_beats_anti_for_a_linear_box():
    box = _LinearBox()
    rng = np.random.default_rng(0)
    Z = pd.DataFrame({"a": rng.normal(size=30), "b": rng.normal(size=30) * 0.1, "c": rng.choice(["x", "y"], 30)})
    ref = pd.Series({"a": 0.0, "b": 0.0, "c": "y"})
    sign = np.ones(30)
    phi = cp.shapley_sampling(box, Z, ref, sign, n_perm=2, seed=0)
    res = cp.deletion_test_e2e(box, Z, ref, sign, {"direct": phi}, k=1, n_random=5, seed=0)
    assert res["direct"]["abs_guided"] > res["abs_random"] > res["direct"]["abs_anti"]


def test_ranking_agreement_of_identical_rankings_is_one():
    a = _toy_timing(5)
    out = cp.ranking_agreement(a, a * 3.0, k=2)
    assert out["spearman_mean"] == pytest.approx(1.0) and out["jaccard_top2_mean"] == 1.0 and out["top1_agreement"] == 1.0


def test_cancellation_index_is_zero_when_channels_agree_and_one_when_they_cancel():
    n, D = 3, 4
    agree = {"channels": {"risk": np.ones((n, D)), "effect_T": np.ones((n, D)), "effect_U": np.zeros((n, D))}}
    assert cp.cancellation(agree)["overall"] == 0.0
    cancel = {"channels": {"risk": np.zeros((n, D)), "effect_T": np.ones((n, D)), "effect_U": -np.ones((n, D))}}
    c = cp.cancellation(cancel)
    assert c["overall"] == 1.0 and np.allclose(c["per_attribute"], 1.0) and np.allclose(c["per_state"], 1.0)


def test_well_defined_counts_degenerate_states():
    phi = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    w = cp.well_defined({"x": phi, "y": np.ones((3, 3))}, ratio=0.05)
    assert w["x"]["n_fallback"] == 2 and w["y"]["n_fallback"] == 0 and w["share_all_defined"] == pytest.approx(1 / 3)


def test_baseline_alignment_of_a_pool_centred_attribution_is_exact():
    rng = np.random.default_rng(0)
    logit = rng.normal(size=50)
    phi = np.zeros((50, 4)); phi[:, 0] = logit - logit.mean()  # sums to the pool-centred distance
    b = cp.baseline_alignment(phi, logit)
    assert abs(b["baseline_gap"]) < 1e-9 and b["sign_agreement"] == 1.0 and b["corr"] == pytest.approx(1.0)


def test_sign_groups_partition_the_attributes():
    per = {"a": {"sign_agreement": 0.95}, "b": {"sign_agreement": 0.1}, "c": {"sign_agreement": 0.5}, "d": {"sign_agreement": None}}
    g = cp.sign_groups(per)
    assert g == {"agree": ["a"], "oppose": ["b"], "independent": ["c"], "one_level_only": ["d"]}
