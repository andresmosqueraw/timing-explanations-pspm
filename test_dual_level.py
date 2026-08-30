"""Smoke tests for dual_level on an untrained SB3 PPO.

Run with the sb3 environment (see requirements.txt), not the repo's .venv:
    pytest experiments/generality_ppo/test_dual_level.py -v
This directory is outside the repo's default pytest testpaths on purpose --
the main suite must stay runnable without stable_baselines3.
"""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

sb3 = pytest.importorskip("stable_baselines3")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dual_level import (  # noqa: E402
    ATTRIBUTION_METHODS,
    CriticHead,
    InterveneLogitHead,
    MarginHead,
    WaitLogitHead,
    deletion_test,
    dual_level_study,
    integrated_gradients,
)


class _ToyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, True, False, {}


@pytest.fixture(scope="module")
def model():
    return sb3.PPO("MlpPolicy", _ToyEnv(), n_steps=32, batch_size=16, seed=0, device="cpu")


@pytest.fixture(scope="module")
def states():
    return np.random.default_rng(0).normal(size=(64, 4)).astype(np.float32)


def test_ig_completeness(model, states):
    """sum_j phi_j must equal f(x) - f(reference) (IG completeness)."""
    head = CriticHead(model.policy)
    ref = states.mean(axis=0)
    phi = integrated_gradients(head, states, ref, n_steps=256)
    with torch.no_grad():
        fx = head(torch.from_numpy(states)).numpy()
        fr = head(torch.from_numpy(ref[None, :])).numpy()[0]
    np.testing.assert_allclose(phi.sum(axis=1), fx - fr, atol=1e-3)


def test_margin_head_matches_action_choice(model, states):
    head = MarginHead(model.policy)
    with torch.no_grad():
        m = head(torch.from_numpy(states)).numpy()
        obs = model.policy.obs_to_tensor(states)[0]
        probs = model.policy.get_distribution(obs).distribution.probs.numpy()
    np.testing.assert_array_equal(m > 0, probs[:, 1] > probs[:, 0])


def test_deletion_test_shapes(model, states):
    head = CriticHead(model.policy)
    ref = states.mean(axis=0)
    phi = integrated_gradients(head, states, ref)
    res = deletion_test(head, states, phi, ref, k=1, n_random=5)
    assert res.abs_guided >= 0 and res.abs_random >= 0 and res.abs_anti >= 0
    assert np.isfinite(res.gap_se)


def test_dual_level_study_runs(model, states):
    out = dual_level_study(model, states, ["a", "b", "c", "d"], ks=(1,), n_margin_min=5, seed=0)
    assert out["n_states"] == 64
    assert "value_test" in out and "phi_v" in out["value_test"]
    if out["margin_evaluable_sample"]:
        r = out["margin_test"]["phi_dq"]["1"]
        assert "flip_guided" in r
        assert "phi_dq_branchwise" in out["margin_test"]
    if out["wait_margin_evaluable_sample"]:
        assert "phi_dq_wait_branchwise" in out["wait_margin_test"]


def test_branchwise_attribution_shape_and_completeness(model, states):
    """phi_intervene - phi_wait must match phi_dq's shape; each branch head
    individually satisfies IG completeness (sum_j phi_j = f(x) - f(ref))."""
    ref = states.mean(axis=0)
    margin_head = MarginHead(model.policy)
    intervene_head = InterveneLogitHead(model.policy)
    wait_head = WaitLogitHead(model.policy)

    phi_dq = integrated_gradients(margin_head, states, ref, n_steps=256)
    phi_intervene = integrated_gradients(intervene_head, states, ref, n_steps=256)
    phi_wait = integrated_gradients(wait_head, states, ref, n_steps=256)
    phi_branchwise = phi_intervene - phi_wait
    assert phi_branchwise.shape == phi_dq.shape

    for head, phi in [(intervene_head, phi_intervene), (wait_head, phi_wait)]:
        with torch.no_grad():
            fx = head(torch.from_numpy(states)).numpy()
            fr = head(torch.from_numpy(ref[None, :])).numpy()[0]
        np.testing.assert_allclose(phi.sum(axis=1), fx - fr, atol=1e-3)


@pytest.mark.parametrize("method_name", list(ATTRIBUTION_METHODS.keys()))
def test_attribution_method_shape_and_finite(model, states, method_name):
    """Each registered method returns a finite phi array matching input shape."""
    fn = ATTRIBUTION_METHODS[method_name]
    head = MarginHead(model.policy)
    ref = states.mean(axis=0)
    small_states = states[:8]  # keep sampling-based methods fast here
    try:
        phi = fn(head, small_states, ref, seed=0)
    except RuntimeError as e:
        if method_name == "deeplift":
            pytest.skip(f"DeepLift hook incompatibility on this policy net: {e}")
        raise
    assert phi.shape == small_states.shape
    assert np.isfinite(phi).all()


def test_linear_methods_branchwise_equals_direct(model, states):
    """Regression guard: IG, Saliency and Occlusion are provably identical
    branch-wise (attribute-then-subtract) vs. direct (subtract-then-attribute)
    -- each is a linear functional of the model output, so subtraction and
    attribution commute. If this ever fails, either the linearity claim in
    the module docstring is wrong or a wrapper stopped matching its method's
    true definition."""
    from dual_level import occlusion_attribution, saliency_attribution

    ref = states.mean(axis=0)
    margin_head = MarginHead(model.policy)
    intervene_head = InterveneLogitHead(model.policy)
    wait_head = WaitLogitHead(model.policy)
    small_states = states[:8]

    for fn, atol in [
        (lambda h, s, r: integrated_gradients(h, s, r, n_steps=256), 1e-4),
        (saliency_attribution, 1e-6),
        (occlusion_attribution, 1e-6),
    ]:
        phi_direct = fn(margin_head, small_states, ref)
        phi_branch = fn(intervene_head, small_states, ref) - fn(wait_head, small_states, ref)
        np.testing.assert_allclose(phi_direct, phi_branch, atol=atol)
