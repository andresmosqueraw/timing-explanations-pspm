"""The explanation methods of the PPM-explainability literature, applied to the
PPO policy's timing decision.

Every function here explains one *target*: a differentiable head over the SB3
policy (``dual_level.WaitMarginHead`` by default, the paper's Delta Q_wait) or,
equivalently, the numpy function ``f(X) -> (N,)`` that ``head_fn`` builds from
it. The methods and the exact library calls mirror the five reference code
bases (see ``run_xai_suite.py`` for the mapping):

* SHAP -- ``shap.KernelExplainer`` (model-agnostic; with 4 features every
  coalition is enumerated, so the values are exact Shapley values for the
  k-means-summarised background), ``shap.DeepExplainer`` and
  ``shap.GradientExplainer`` (the neural-network counterparts of the
  ``shap.TreeExplainer`` the reference repos apply to XGBoost).
* LIME -- ``lime_stability.LimeTabularExplainerOvr`` (Elkhawaga et al.'s
  choice), in regression mode on the target, with the CSI/VSI stability
  indices of Visani et al. available through ``lime_stability``.
* Permutation importance -- ``n_repeats`` shuffles of one column; the
  policy has no label to score against, so the "score decrease" is the
  root-mean-square displacement of the target (the same replacement the DL
  branch of Stevens & De Smedt uses).
* Observed-value replacement importance (Stevens & De Smedt's
  ``var_importance``): every cell of a column replaced by another value
  observed in that column.
* ALE (Apley & Zhu), the one-feature accumulated local effects that
  Elkhawaga et al. compute with ``alibi``; implemented here directly (alibi
  does not install on this interpreter) with the same quantile grid.
* Integrated Gradients -- the paper's own method, from ``dual_level``, so
  that it can be scored with the same metrics as the others.

All attribution functions return ``phi`` with ``phi.shape == states.shape``
(local attributions) so that the metrics in ``xai_metrics`` and the deletion
test in ``dual_level`` accept any of them interchangeably.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from dual_level import (
    CriticHead,
    MarginHead,
    WaitMarginHead,
    integrated_gradients,
)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

TARGETS = ("wait_margin", "margin", "p_wait", "p_intervene", "value")


class _ProbHead(nn.Module):
    """pi(action | s) as a differentiable function of the observation."""

    def __init__(self, policy, action: int):
        super().__init__()
        self.policy = policy
        self.action = action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self.policy.get_distribution(obs)
        return dist.distribution.probs[..., self.action]


def build_head(policy, target: str = "wait_margin", intervene_action: int = 1) -> nn.Module:
    """The differentiable head for ``target`` (one of ``TARGETS``)."""
    if target == "wait_margin":
        return WaitMarginHead(MarginHead(policy, intervene_action))
    if target == "margin":
        return MarginHead(policy, intervene_action)
    if target == "p_wait":
        return _ProbHead(policy, 1 - intervene_action)
    if target == "p_intervene":
        return _ProbHead(policy, intervene_action)
    if target == "value":
        return CriticHead(policy)
    raise ValueError(f"unknown target {target!r}; choose one of {TARGETS}")


def head_fn(head: nn.Module):
    """``f(X: array (N, D)) -> array (N,)``, the black box the model-agnostic
    methods (KernelSHAP, LIME, permutation, ALE) call."""
    dev = next(head.policy.parameters()).device

    def f(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            return head(torch.from_numpy(X).to(dev)).cpu().numpy().astype(np.float64)

    return f


class _Column(nn.Module):
    """``shap``'s PyTorch explainers want an (N, 1) output, not (N,)."""

    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class Attribution:
    """Local attributions of one method on one pool of states."""

    method: str
    phi: np.ndarray  # (N, D)
    expected_value: float | None = None  # SHAP base value / IG reference output
    seconds: float = 0.0
    extras: dict = field(default_factory=dict)

    @property
    def global_importance(self) -> np.ndarray:
        """Mean |phi| per feature -- the global importance every reference
        repo derives from local SHAP values (``shap_values.abs.mean(0)``)."""
        return np.abs(self.phi).mean(axis=0)


# ---------------------------------------------------------------------------
# SHAP (the shap library)
# ---------------------------------------------------------------------------


def kernel_shap(head: nn.Module, states: np.ndarray, background: np.ndarray, n_background: int = 50, seed: int = 123) -> Attribution:
    """``shap.KernelExplainer`` on the target, background summarised with
    ``shap.kmeans`` (weighted centroids). ``nsamples="auto"`` enumerates all
    2^D coalitions for D=4, so the result is exact for that background."""
    import shap

    t0 = time.perf_counter()
    f = head_fn(head)
    np.random.seed(seed)
    bg = shap.kmeans(background, min(n_background, len(background)))
    explainer = shap.KernelExplainer(f, bg)
    phi = np.asarray(explainer.shap_values(states, silent=True), dtype=np.float64)
    return Attribution("kernel_shap", phi, float(np.ravel(explainer.expected_value)[0]), time.perf_counter() - t0)


def deep_shap(head: nn.Module, states: np.ndarray, background: np.ndarray, n_background: int = 100, seed: int = 123) -> Attribution:
    """``shap.DeepExplainer`` (DeepLIFT-style SHAP) on the PyTorch head."""
    import shap

    t0 = time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(background), size=min(n_background, len(background)), replace=False)
    bg = torch.from_numpy(background[idx].astype(np.float32))
    explainer = shap.DeepExplainer(_Column(head), bg)
    sv = explainer.shap_values(torch.from_numpy(states.astype(np.float32)), check_additivity=False)
    phi = np.asarray(sv, dtype=np.float64).reshape(len(states), states.shape[1])
    ev = float(np.ravel(np.asarray(explainer.expected_value))[0])
    return Attribution("deep_shap", phi, ev, time.perf_counter() - t0)


def gradient_shap(head: nn.Module, states: np.ndarray, background: np.ndarray, n_background: int = 100, nsamples: int = 200, seed: int = 123) -> Attribution:
    """``shap.GradientExplainer`` (expected gradients) on the PyTorch head."""
    import shap

    t0 = time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(background), size=min(n_background, len(background)), replace=False)
    bg = torch.from_numpy(background[idx].astype(np.float32))
    explainer = shap.GradientExplainer(_Column(head), bg)
    sv = explainer.shap_values(torch.from_numpy(states.astype(np.float32)), nsamples=nsamples)
    phi = np.asarray(sv, dtype=np.float64).reshape(len(states), states.shape[1])
    ev = float(head_fn(head)(background[idx]).mean())
    return Attribution("gradient_shap", phi, ev, time.perf_counter() - t0)


def shap_explanation(att: Attribution, states: np.ndarray, feature_names: list[str]):
    """Wrap an Attribution as a ``shap.Explanation`` so the shap plotting
    API (beeswarm, bar, waterfall, scatter, ...) accepts it."""
    import shap

    base = np.full(len(states), att.expected_value if att.expected_value is not None else 0.0)
    return shap.Explanation(values=att.phi, base_values=base, data=states.astype(np.float64), feature_names=list(feature_names))


# ---------------------------------------------------------------------------
# LIME (lime + lime_stability)
# ---------------------------------------------------------------------------


def lime_explainer(background: np.ndarray, feature_names: list[str], discretize: bool = False, kernel_width: float | None = None, seed: int = 123):
    """``LimeTabularExplainerOvr`` (lime_stability's subclass of
    ``LimeTabularExplainer``) in regression mode on the target.

    ``discretize_continuous`` is LIME's default (and Elkhawaga et al.'s), but
    on this policy the discretised surrogate fits some states very poorly
    (local R^2 below 0.01 on states near a bin edge of
    ``available_resources``), so the suite defaults to the continuous
    surrogate and records every local R^2 (``lime_local_r2``) either way.
    """
    from lime_stability.stability import LimeTabularExplainerOvr

    kwargs = dict(
        training_data=np.asarray(background, dtype=np.float64),
        feature_names=list(feature_names),
        mode="regression",
        discretize_continuous=discretize,
        random_state=seed,
    )
    if kernel_width is not None:
        kwargs["kernel_width"] = kernel_width
    return LimeTabularExplainerOvr(**kwargs)


def lime_attribution(head: nn.Module, states: np.ndarray, background: np.ndarray, feature_names: list[str], num_samples: int = 5000, discretize: bool = False, seed: int = 123) -> Attribution:
    """LIME coefficients of every state (all D features kept), plus the local
    R^2 and intercept of each surrogate."""
    t0 = time.perf_counter()
    f = head_fn(head)
    explainer = lime_explainer(background, feature_names, discretize=discretize, seed=seed)
    D = states.shape[1]
    phi = np.zeros((len(states), D))
    r2 = np.zeros(len(states))
    intercept = np.zeros(len(states))
    for i, s in enumerate(states):
        exp = explainer.explain_instance(np.asarray(s, dtype=np.float64), f, num_features=D, num_samples=num_samples)
        label = 1 if 1 in exp.local_exp else 0
        for j, w in exp.local_exp[label]:
            phi[i, j] = w
        r2[i] = exp.score
        intercept[i] = float(exp.intercept[label])
    return Attribution(
        "lime", phi, None, time.perf_counter() - t0,
        extras={"lime_local_r2": r2.tolist(), "lime_local_r2_mean": float(r2.mean()), "lime_intercept": intercept.tolist(),
                "discretize_continuous": bool(discretize), "num_samples": int(num_samples)},
    )


def lime_stability(head: nn.Module, state: np.ndarray, background: np.ndarray, feature_names: list[str], n_calls: int = 10, num_samples: int = 5000, discretize: bool = False, seed: int = 123) -> dict:
    """Visani et al.'s CSI / VSI for one state: ``n_calls`` repeated LIME
    explanations; VSI = agreement of the selected-variable sets, CSI =
    overlap of the coefficients' confidence intervals, both in [0, 100]
    (``LimeTabularExplainerOvr.check_stability``, as Elkhawaga et al.)."""
    f = head_fn(head)
    explainer = lime_explainer(background, feature_names, discretize=discretize, seed=seed)
    csi, vsi = explainer.check_stability(np.asarray(state, dtype=np.float64), f, num_features=len(feature_names), num_samples=num_samples, n_calls=n_calls)
    return {"csi": float(csi), "vsi": float(vsi), "n_calls": int(n_calls)}


# ---------------------------------------------------------------------------
# Permutation-style global importances
# ---------------------------------------------------------------------------


def permutation_importance(head: nn.Module, states: np.ndarray, n_repeats: int = 10, seed: int = 42) -> dict:
    """``sklearn.inspection.permutation_importance`` translated to a label-free
    target: shuffle column j across the pool, score = RMS displacement of the
    target. Returns mean, std and the ``n_repeats`` individual repetitions
    (the reference repos box-plot those and compare them across models)."""
    t0 = time.perf_counter()
    f = head_fn(head)
    rng = np.random.default_rng(seed)
    f0 = f(states)
    D = states.shape[1]
    reps = np.zeros((D, n_repeats))
    for j in range(D):
        for r in range(n_repeats):
            perm = states.copy()
            perm[:, j] = rng.permutation(perm[:, j])
            reps[j, r] = float(np.sqrt(np.mean((f(perm) - f0) ** 2)))
    return {
        "importances_mean": reps.mean(axis=1).tolist(),
        "importances_std": reps.std(axis=1, ddof=0).tolist(),
        "importances": reps.tolist(),
        "n_repeats": int(n_repeats),
        "seconds": time.perf_counter() - t0,
    }


def replace_with_other_observed(column: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Each cell replaced by a value observed elsewhere in the column, never
    its own (Stevens & De Smedt's ``np.setdiff1d(domain, [value])`` +
    ``random.choice``); a constant column is returned unchanged."""
    domain = np.unique(column)
    if len(domain) < 2:
        return column.copy()
    out = np.empty_like(column)
    for i, v in enumerate(column):
        others = domain[domain != v]
        out[i] = rng.choice(others)
    return out


def var_importance(head: nn.Module, states: np.ndarray, seed: int = 42) -> dict:
    """Stevens & De Smedt's ``var_importance``: the per-feature effect of
    replacing every cell of column j by another observed value. Their ML
    branch measures the MSE increase against the label; a policy has no
    label, so the effect is the RMS displacement of the target (their DL
    branch's ``((orig - perturbed)**2).mean()**0.5``). Also reports the
    argmax-action flip rate under the same replacement."""
    t0 = time.perf_counter()
    f = head_fn(head)
    rng = np.random.default_rng(seed)
    f0 = f(states)
    D = states.shape[1]
    effects, flips, l1 = np.zeros(D), np.zeros(D), np.zeros(D)
    for j in range(D):
        pert = states.copy()
        pert[:, j] = replace_with_other_observed(states[:, j], rng)
        fj = f(pert)
        effects[j] = float(np.sqrt(np.mean((fj - f0) ** 2)))
        l1[j] = float(np.mean(np.abs(fj - f0)))
        flips[j] = float(np.mean(np.sign(fj) != np.sign(f0)))
    return {"effects": effects.tolist(), "l1_shift": l1.tolist(), "sign_flip_rate": flips.tolist(), "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------------------
# ALE (Apley & Zhu), one feature at a time
# ---------------------------------------------------------------------------


def ale_1d(head: nn.Module, states: np.ndarray, feature: int, n_bins: int = 10) -> dict:
    """Accumulated local effects of one feature on the target, on the
    quantile grid alibi uses (``min_bin_points``-free version): grid =
    quantiles of the feature (or its distinct values when there are fewer
    than ``n_bins``), local effect in bin k = mean over the states in the
    bin of f(x with feature at the upper edge) - f(x with feature at the
    lower edge), accumulated and centred to zero mean over the pool."""
    f = head_fn(head)
    col = states[:, feature].astype(np.float64)
    uniq = np.unique(col)
    if len(uniq) <= n_bins:
        grid = uniq
    else:
        grid = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))
    if len(grid) < 2:
        return {"grid": grid.tolist(), "ale": [0.0] * len(grid), "counts": [int(len(col))]}
    bins = np.clip(np.searchsorted(grid, col, side="left") - 1, 0, len(grid) - 2)
    local = np.zeros(len(grid) - 1)
    counts = np.zeros(len(grid) - 1, dtype=int)
    for k in range(len(grid) - 1):
        members = states[bins == k]
        counts[k] = len(members)
        if len(members) == 0:
            continue
        lo, hi = members.copy(), members.copy()
        lo[:, feature], hi[:, feature] = grid[k], grid[k + 1]
        local[k] = float(np.mean(f(hi) - f(lo)))
    acc = np.concatenate([[0.0], np.cumsum(local)])
    # centre: weighted mean over the bins (each bin's ALE = average of its two edges)
    w = counts / max(counts.sum(), 1)
    centre = float(np.sum(w * (acc[:-1] + acc[1:]) / 2.0))
    return {"grid": grid.tolist(), "ale": (acc - centre).tolist(), "counts": counts.tolist()}


def ale_all(head: nn.Module, states: np.ndarray, feature_names: list[str], n_bins: int = 10) -> dict:
    t0 = time.perf_counter()
    out = {name: ale_1d(head, states, j, n_bins) for j, name in enumerate(feature_names)}
    out["seconds"] = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------
# The paper's own method, for comparison under the same metrics
# ---------------------------------------------------------------------------


def ig_attribution(head: nn.Module, states: np.ndarray, reference: np.ndarray, n_steps: int = 128) -> Attribution:
    t0 = time.perf_counter()
    phi = integrated_gradients(head, states, reference, n_steps=n_steps).astype(np.float64)
    ev = float(head_fn(head)(reference[None, :])[0])
    return Attribution("integrated_gradients", phi, ev, time.perf_counter() - t0)
