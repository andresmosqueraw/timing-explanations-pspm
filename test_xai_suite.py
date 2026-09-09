"""Unit tests for xai_methods / xai_metrics on functions with known answers.

Run with the repo's .venv (shap, lime, lime_stability, statsmodels installed):
    pytest test_xai_suite.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import xai_metrics as xm
import xai_methods as xme

FEATS = ["relative_position", "reliability", "deviation", "available_resources"]


class _LinearHead(nn.Module):
    """f(x) = w . x + b, with ``policy`` so head_fn / _device find a device."""

    def __init__(self, w, b=0.0):
        super().__init__()
        self.lin = nn.Linear(len(w), 1)
        with torch.no_grad():
            self.lin.weight.copy_(torch.tensor([w], dtype=torch.float32))
            self.lin.bias.fill_(b)
        self.policy = self.lin  # duck-typing what dual_level/xai_methods expect

    def forward(self, x):
        return self.lin(x).squeeze(-1)


@pytest.fixture
def linear():
    return _LinearHead([0.5, 0.0, -2.0, 4.0], 1.0)


@pytest.fixture
def states():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(120, 4)).astype(np.float32)
    X[:, 3] = rng.integers(0, 4, size=120)
    return X


# --- methods ----------------------------------------------------------------


def test_kernel_shap_is_exact_on_linear(linear, states):
    att = xme.kernel_shap(linear, states[:20], states, seed=1)
    f = xme.head_fn(linear)
    # completeness: sum(phi) = f(x) - E[f(background)]
    np.testing.assert_allclose(att.phi.sum(axis=1), f(states[:20]) - att.expected_value, atol=1e-4)
    # a zero-weight feature gets zero attribution
    assert np.abs(att.phi[:, 1]).max() < 1e-6


def test_deep_and_gradient_shap_match_kernel_on_linear(linear, states):
    ref = xme.kernel_shap(linear, states[:10], states, seed=1).global_importance
    for fn in (xme.deep_shap, xme.gradient_shap):
        att = fn(linear, states[:10], states, seed=1)
        assert att.phi.shape == (10, 4)
        assert np.argmax(att.global_importance) == np.argmax(ref) == 3


def test_lime_recovers_linear_coefficients(linear, states):
    att = xme.lime_attribution(linear, states[:5], states, FEATS, num_samples=2000, seed=1)
    assert att.phi.shape == (5, 4)
    assert att.extras["lime_local_r2_mean"] > 0.99
    # signs of the coefficients follow the true weights
    assert (att.phi[:, 3] > 0).all() and (att.phi[:, 2] < 0).all()


def test_lime_stability_indices_in_range(linear, states):
    st = xme.lime_stability(linear, states[0], states, FEATS, n_calls=3, num_samples=500, seed=1)
    assert 0 <= st["csi"] <= 100 and 0 <= st["vsi"] <= 100


def test_permutation_and_var_importance_rank_true_weights(linear, states):
    perm = xme.permutation_importance(linear, states, n_repeats=3, seed=1)
    assert np.argmax(perm["importances_mean"]) == 3
    assert perm["importances_mean"][1] == pytest.approx(0.0, abs=1e-6)
    v = xme.var_importance(linear, states, seed=1)
    assert np.argmax(v["effects"]) == 3 and v["effects"][1] == pytest.approx(0.0, abs=1e-6)


def test_replace_with_other_observed_never_keeps_value():
    rng = np.random.default_rng(0)
    col = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0])
    out = xme.replace_with_other_observed(col, rng)
    assert (out != col).all() and set(out) <= set(col)
    const = np.ones(5)
    np.testing.assert_array_equal(xme.replace_with_other_observed(const, rng), const)


def test_ale_of_linear_is_linear(linear, states):
    a = xme.ale_1d(linear, states, feature=3)
    grid, ale = np.array(a["grid"]), np.array(a["ale"])
    slopes = np.diff(ale) / np.diff(grid)
    np.testing.assert_allclose(slopes, 4.0, atol=1e-4)
    # centred: weighted mean over bins is ~0
    assert abs(np.average((ale[:-1] + ale[1:]) / 2, weights=a["counts"])) < 1e-6


# --- metrics ----------------------------------------------------------------


def test_parsimony_counts():
    p = xm.parsimony(np.array([0.9, 0.0, 0.05, 0.05]), FEATS, share_threshold=0.05)
    assert p["nonzero"] == 3 and p["above_share"] == 3
    assert p["by_family"] == {"progress": 1, "prediction": 1, "capacity": 1, "effect": 0}
    assert 1.0 < p["effective"] < 3.0


def test_functional_complexity_on_sign_function(states):
    f = lambda X: np.sign(np.asarray(X)[:, 3] - 1.5)  # flips only through feature 3
    fc = xm.functional_complexity(f, states, FEATS, seed=0)
    assert fc["per_feature"]["reliability"]["flip"] == 0.0
    assert fc["per_feature"]["available_resources"]["flip"] > 0.3
    assert fc["per_family"]["capacity"]["l1"] == pytest.approx(200 * fc["per_family"]["capacity"]["flip"])


def test_monotonicity_and_lod():
    g = np.array([0.1, 0.2, 0.3, 0.4])
    m = xm.monotonicity(g, g * 3)
    assert m["spearman"] == pytest.approx(1.0) and m["kendall"] == pytest.approx(1.0)
    assert xm.monotonicity(g, -g)["spearman"] == pytest.approx(-1.0)
    l = xm.lod(g, g, FEATS, k=2)
    assert l["lod"] == 0.0 and l["normalised_distance"] == pytest.approx(0.0)
    # top-2 {deviation, available_resources} vs {relative_position, reliability}
    l2 = xm.lod(g, g[::-1], FEATS, k=2)
    # families: progress, prediction, capacity, effect (the last only in the cate variants)
    assert l2["explanation_counts"] == [0, 1, 1, 0] and l2["model_counts"] == [1, 1, 0, 0]
    assert l2["lod"] == pytest.approx(np.sqrt(2))


def test_consisxai_reducts_core_ratio_and_abic():
    crit = {
        "a": [0.0, 0.1, 0.2, 1.0],
        "b": [0.0, 0.0, 0.9, 1.0],
        "c": [0.1, 0.0, 0.0, 1.0],
    }
    rc = xm.reducts_and_core(crit, FEATS, seed=0)
    assert rc["core"] == ["available_resources"]
    assert "available_resources" in rc["selected_reduct"]
    res = xm.consisxai(np.array([0.0, 0.0, 0.3, 0.7]), FEATS, rc, n=100)
    assert res["core"]["experimental"] == 1.0 and res["core"]["regular"] == 1.0
    assert res["core"]["abic_regular"]["aic"] is None  # ratio 1 -> log undefined
    # a wrong explanation misses the core
    bad = xm.consisxai(np.array([1.0, 0.0, 0.0, 0.0]), FEATS, rc, n=100)
    assert bad["core"]["experimental"] == 0.0 and bad["core"]["missed"] == 1
    assert bad["core"]["abic_experimental"]["aic"] == pytest.approx(2.0)  # -2 log2(1) + 2*1
    assert bad["core"]["abic_experimental"]["bic"] == pytest.approx(np.log2(100))


def test_agreement_and_rediscovery():
    ag = xm.agreement({"x": np.array([1, 2, 3, 4.0]), "y": np.array([4, 3, 2, 1.0])}, FEATS, k=2)
    p = ag["pairs"]["x|y"]
    assert p["spearman"] == pytest.approx(-1.0) and not p["top1_same"] and p["jaccard_top2"] == 0.0
    r = xm.rediscovery_rate(np.array([0.9, 0.0, 0.0, 0.1]), FEATS, ["available_resources", "reliability"])
    assert r["rate"] == 0.5 and r["matches"] == ["available_resources"]


def test_build_head_targets_on_untrained_ppo():
    gym = pytest.importorskip("gymnasium")
    from gymnasium import spaces
    from stable_baselines3 import PPO

    class Env(gym.Env):
        observation_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        action_space = spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            return np.zeros(4, np.float32), {}

        def step(self, a):
            return np.zeros(4, np.float32), 0.0, True, False, {}

    model = PPO("MlpPolicy", Env(), n_steps=8, batch_size=8, seed=0, verbose=0)
    X = np.random.default_rng(0).normal(size=(6, 4)).astype(np.float32)
    outs = {t: xme.head_fn(xme.build_head(model.policy, t))(X) for t in xme.TARGETS}
    np.testing.assert_allclose(outs["wait_margin"], -outs["margin"], atol=1e-6)
    np.testing.assert_allclose(outs["p_wait"] + outs["p_intervene"], 1.0, atol=1e-6)
    np.testing.assert_allclose(np.log(outs["p_intervene"] / outs["p_wait"]), outs["margin"], atol=1e-5)
