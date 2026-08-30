"""Dual-level attribution (PL-xPsPM) instantiated on SB3 PPO agents.

Ports the paper's two explanation targets from the TDQN sequence encoder to a
feature-vector PPO policy:

- **Risk** (*why is this case at risk?*): Integrated Gradients on the critic
  V(s). Same semantics as the paper's phi^V (completeness w.r.t. the value).
- **Margin** (*why act now rather than wait?*): Integrated Gradients on the
  actor's logit margin log pi(intervene|s) - log pi(wait|s), scored on the
  states where the policy's argmax action is "intervene". This is a
  *declared semantics change* versus the paper's Delta-Q: a policy-preference
  margin, not an expected-return difference (the paper's Def. 2 states the
  questions independently of how the targets are computed).
- **Wait-side margin** (*why wait rather than act now?*): the same margin
  with the sign flipped, log pi(wait|s) - log pi(intervene|s), scored on the
  states where the policy's argmax action is "wait" instead. For an agent
  that rarely or never intervenes, this is what makes a timing
  justification evaluable at all — the paper's Delta-Q_wait, used for TDQN
  on Sepsis.

- **Branch-wise margin** (Padella et al. 2022 style): the same margin target,
  but attributed by computing phi(logit(intervene)) and phi(logit(wait))
  *separately* via Integrated Gradients and subtracting, rather than
  attributing the margin function directly (MarginHead/WaitMarginHead
  above). Padella et al. explain a next-activity recommendation by taking
  SHAP of the KPI prediction before and after the recommended activity and
  subtracting; here there is no literal before/after state (our features
  do not change with the chosen action), so the two "states" are the two
  action branches of the same state. IG is not linear, so this
  attribute-then-subtract ranking can disagree with the direct
  subtract-then-attribute one -- both are scored against the same margin
  function, so any disagreement is attributable to the attribution method
  alone.

- **Multi-method before/after comparison**: the branch-wise-vs-direct check
  above, repeated for Saliency, DeepLift, GradientShap, Occlusion
  (FeatureAblation), ShapleyValueSampling, KernelShap and Lime (all via
  Captum), not just Integrated Gradients. IG, Saliency and Occlusion are
  provably identical branch-wise vs. direct (each is a linear functional of
  the model output); the sampling/fitting-based methods (GradientShap,
  ShapleyValueSampling, KernelShap, Lime) and DeepLift's rule-based backprop
  are not guaranteed to agree, so this is where real divergence, if any,
  shows up. See ATTRIBUTION_METHODS / _multi_method_comparison.

The fidelity tests are the paper's deletion tests ported to feature space:
masking replaces a feature with a *reference value* (the mean over the
background states, the analog of the PAD baseline), and guided / random /
anti-guided |displacement| of the target are compared. The sign-flip rate of
the margin is reported alongside, as in Def. 4's refinement. The evaluability
criterion drops the masking-granularity condition (meaningless for a handful
of named features) and keeps the sample and target-above-null conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------
# Differentiable heads over an SB3 ActorCriticPolicy
# --------------------------------------------------------------------------


def _latents(policy, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(latent_pi, latent_vf) via the policy's own extractors, differentiable."""
    features = policy.extract_features(obs)
    if isinstance(features, tuple):
        pi_features, vf_features = features
        latent_pi = policy.mlp_extractor.forward_actor(pi_features)
        latent_vf = policy.mlp_extractor.forward_critic(vf_features)
    else:
        latent_pi, latent_vf = policy.mlp_extractor(features)
    return latent_pi, latent_vf


class CriticHead(nn.Module):
    """V(s) as a differentiable function of the observation."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        _, latent_vf = _latents(self.policy, obs)
        return self.policy.value_net(latent_vf).squeeze(-1)


class MarginHead(nn.Module):
    """logit(intervene) - logit(wait) as a differentiable function of obs."""

    def __init__(self, policy, intervene_action: int = 1):
        super().__init__()
        self.policy = policy
        self.intervene_action = intervene_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent_pi, _ = _latents(self.policy, obs)
        logits = self.policy.action_net(latent_pi)
        a = self.intervene_action
        wait = 1 - a if logits.shape[-1] == 2 else 0
        return logits[..., a] - logits[..., wait]


class InterveneLogitHead(nn.Module):
    """logit(intervene) alone, as a differentiable function of obs.

    Paired with WaitLogitHead to build a branch-wise attribution
    phi(logit(intervene)) - phi(logit(wait)) -- attribute-then-subtract,
    mirroring Padella et al. 2022's before/after-recommendation SHAP delta,
    translated to two branches of one state rather than two sequential
    states (our features do not change with the chosen action). Contrasts
    with MarginHead's direct, subtract-then-attribute IG on the margin
    function itself; IG is not linear, so the two can disagree.
    """

    def __init__(self, policy, intervene_action: int = 1):
        super().__init__()
        self.policy = policy
        self.intervene_action = intervene_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent_pi, _ = _latents(self.policy, obs)
        logits = self.policy.action_net(latent_pi)
        return logits[..., self.intervene_action]


class WaitLogitHead(nn.Module):
    """logit(wait) alone, as a differentiable function of obs. See InterveneLogitHead."""

    def __init__(self, policy, intervene_action: int = 1):
        super().__init__()
        self.policy = policy
        self.intervene_action = intervene_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent_pi, _ = _latents(self.policy, obs)
        logits = self.policy.action_net(latent_pi)
        a = self.intervene_action
        wait = 1 - a if logits.shape[-1] == 2 else 0
        return logits[..., wait]


class WaitMarginHead(nn.Module):
    """Delta Q_wait(s) = logit(wait) - logit(intervene) = -MarginHead(s).

    The timing *justification for waiting* rather than for acting now — the
    paper's wait-side margin (used for TDQN on states where the policy
    waits). Non-negative on the states where the policy's argmax action is
    "wait", by construction.
    """

    def __init__(self, margin_head: "MarginHead"):
        super().__init__()
        self.margin_head = margin_head
        self.policy = margin_head.policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return -self.margin_head(obs)


# --------------------------------------------------------------------------
# Integrated Gradients (feature space, straight-line path from reference)
# --------------------------------------------------------------------------


def _device(head: nn.Module) -> torch.device:
    return next(head.policy.parameters()).device


def integrated_gradients(
    head: nn.Module,
    states: np.ndarray,
    reference: np.ndarray,
    n_steps: int = 128,
) -> np.ndarray:
    """IG attributions phi with sum_j phi_j = f(x) - f(reference), per state."""
    dev = _device(head)
    x = torch.from_numpy(states.astype(np.float32)).to(dev)
    ref = torch.from_numpy(reference.astype(np.float32)).to(dev).expand_as(x)
    alphas = torch.linspace(0.0, 1.0, n_steps, device=dev).view(-1, 1, 1)
    path = ref.unsqueeze(0) + alphas * (x - ref).unsqueeze(0)  # (S, N, D)
    path = path.reshape(-1, x.shape[1]).requires_grad_(True)
    out = head(path)
    grads = torch.autograd.grad(out.sum(), path)[0]
    grads = grads.reshape(n_steps, x.shape[0], x.shape[1]).mean(dim=0)
    phi = (x - ref) * grads
    return phi.detach().cpu().numpy()


# --------------------------------------------------------------------------
# Additional attribution methods (Captum), for the multi-method before/after
# comparison: does attribute-then-subtract (branch-wise) agree with
# subtract-then-attribute (direct) for methods other than Integrated
# Gradients? IG, Saliency and Occlusion/FeatureAblation are provably
# identical either way (each is a linear functional of the model output, so
# subtraction commutes with attribution); DeepLift, GradientShap,
# ShapleyValueSampling, KernelShap and Lime are NOT guaranteed to agree
# (sampling/fitting-based, or rule-based backprop that isn't integration) --
# real divergence, if any, shows up only in that second group. Every wrapper
# returns phi: np.ndarray matching states.shape, the same contract
# integrated_gradients uses, so deletion_test needs no changes.
# --------------------------------------------------------------------------

from captum.attr import (
    DeepLift,
    FeatureAblation,
    GradientShap,
    KernelShap,
    Lime,
    Saliency,
    ShapleyValueSampling,
)


def _prep(head: nn.Module, states: np.ndarray, reference: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    dev = _device(head)
    x = torch.from_numpy(states.astype(np.float32)).to(dev)
    ref = torch.from_numpy(reference.astype(np.float32)).to(dev).expand_as(x)
    return x, ref


def saliency_attribution(head: nn.Module, states: np.ndarray, reference: np.ndarray, **_) -> np.ndarray:
    x, _ref = _prep(head, states, reference)
    x = x.clone().requires_grad_(True)
    phi = Saliency(head).attribute(x, abs=False)
    return phi.detach().cpu().numpy()


def deeplift_attribution(head: nn.Module, states: np.ndarray, reference: np.ndarray, **_) -> np.ndarray:
    x, ref = _prep(head, states, reference)
    x = x.clone().requires_grad_(True)
    phi = DeepLift(head).attribute(x, baselines=ref)
    return phi.detach().cpu().numpy()


def gradientshap_attribution(
    head: nn.Module, states: np.ndarray, reference: np.ndarray, n_samples: int = 20, stdevs: float = 0.1, seed: int = 123, **_
) -> np.ndarray:
    x, ref = _prep(head, states, reference)
    x = x.clone().requires_grad_(True)
    torch.manual_seed(seed)
    phi = GradientShap(head).attribute(x, baselines=ref, n_samples=n_samples, stdevs=stdevs)
    return phi.detach().cpu().numpy()


def occlusion_attribution(head: nn.Module, states: np.ndarray, reference: np.ndarray, **_) -> np.ndarray:
    # FeatureAblation: the tabular analog of image Occlusion -- ablates one
    # feature at a time, no contiguous "window" needed for a flat vector.
    x, ref = _prep(head, states, reference)
    phi = FeatureAblation(head).attribute(x, baselines=ref)
    return phi.detach().cpu().numpy()


def shapley_sampling_attribution(
    head: nn.Module, states: np.ndarray, reference: np.ndarray, n_samples: int = 30, seed: int = 123, **_
) -> np.ndarray:
    x, ref = _prep(head, states, reference)
    torch.manual_seed(seed)
    phi = ShapleyValueSampling(head).attribute(x, baselines=ref, n_samples=n_samples, perturbations_per_eval=64)
    return phi.detach().cpu().numpy()


def kernelshap_attribution(
    head: nn.Module, states: np.ndarray, reference: np.ndarray, n_samples: int = 50, seed: int = 123, **_
) -> np.ndarray:
    x, ref = _prep(head, states, reference)
    torch.manual_seed(seed)
    phi = KernelShap(head).attribute(x, baselines=ref, n_samples=n_samples, perturbations_per_eval=64)
    return phi.detach().cpu().numpy()


def lime_attribution(
    head: nn.Module, states: np.ndarray, reference: np.ndarray, n_samples: int = 50, seed: int = 123, **_
) -> np.ndarray:
    x, ref = _prep(head, states, reference)
    torch.manual_seed(seed)
    phi = Lime(head).attribute(x, baselines=ref, n_samples=n_samples, perturbations_per_eval=64)
    return phi.detach().cpu().numpy()


ATTRIBUTION_METHODS: dict = {
    "saliency": saliency_attribution,
    "deeplift": deeplift_attribution,
    "gradientshap": gradientshap_attribution,
    "occlusion": occlusion_attribution,
    "shapley_sampling": shapley_sampling_attribution,
    "kernelshap": kernelshap_attribution,
    "lime": lime_attribution,
}

# States used for the (expensive) multi-method comparison are capped well
# below max_margin_states -- sampling/fitting methods cost O(n_samples x
# n_states) forward passes per attribution call, x3 calls (direct, intervene
# branch, wait branch) x7 methods x2 sides. n_samples above are chosen small
# for the same reason (tens, not hundreds/thousands) -- a 3-4 dim state
# needs far fewer perturbation samples to converge than a real-world
# high-dimensional input.
MULTI_METHOD_MAX_STATES = 100


def _multi_method_comparison(
    scoring_head: nn.Module,
    intervene_head: nn.Module,
    wait_head: nn.Module,
    states: np.ndarray,
    reference: np.ndarray,
    ks: tuple[int, ...],
    seed: int,
    branch_sign: float = 1.0,
) -> dict:
    """Run the before/after (branch-wise vs. direct) comparison across
    ATTRIBUTION_METHODS, scored against ``scoring_head`` (the actual margin
    function -- only the attribution differs between direct and branchwise,
    not what is measured). ``branch_sign`` flips the branch subtraction for
    the wait side (wait - intervene instead of intervene - wait).
    """
    if len(states) > MULTI_METHOD_MAX_STATES:
        idx = np.random.default_rng(seed).choice(len(states), size=MULTI_METHOD_MAX_STATES, replace=False)
        states = states[idx]
    out: dict = {}
    for name, fn in ATTRIBUTION_METHODS.items():
        try:
            phi_direct = fn(scoring_head, states, reference, seed=seed)
            phi_i = fn(intervene_head, states, reference, seed=seed)
            phi_w = fn(wait_head, states, reference, seed=seed)
        except RuntimeError as e:
            # Some Captum methods (notably DeepLift's hook-based backward)
            # are architecture-sensitive and can fail on a shared-submodule
            # policy network without indicating a bug in this harness --
            # record the failure rather than aborting the whole comparison.
            out[name] = {"error": str(e)}
            continue
        phi_branch = branch_sign * (phi_i - phi_w)
        per_k = {}
        pass_direct_any, pass_branch_any = False, False
        for k in ks:
            res_direct = deletion_test(scoring_head, states, phi_direct, reference, k, seed=seed, track_sign_flips=True)
            res_branch = deletion_test(scoring_head, states, phi_branch, reference, k, seed=seed, track_sign_flips=True)
            z_direct = res_direct.gap / res_direct.gap_se if res_direct.gap_se > 0 else float("nan")
            z_branch = res_branch.gap / res_branch.gap_se if res_branch.gap_se > 0 else float("nan")
            pass_direct = bool(np.isfinite(z_direct) and z_direct >= 3.0)
            pass_branch = bool(np.isfinite(z_branch) and z_branch >= 3.0)
            pass_direct_any = pass_direct_any or pass_direct
            pass_branch_any = pass_branch_any or pass_branch
            per_k[str(k)] = {
                "direct": res_direct.as_dict(),
                "branchwise": res_branch.as_dict(),
                "z_direct": float(z_direct),
                "z_branchwise": float(z_branch),
                "pass_direct": pass_direct,
                "pass_branchwise": pass_branch,
            }
        out[name] = {
            "n_states": int(len(states)),
            "by_k": per_k,
            "max_abs_diff": float(np.abs(phi_direct - phi_branch).max()),
            "mean_abs_diff": float(np.abs(phi_direct - phi_branch).mean()),
            "same_verdict": bool(pass_direct_any == pass_branch_any),
        }
    return out


# --------------------------------------------------------------------------
# Feature-deletion fidelity test
# --------------------------------------------------------------------------


@dataclass
class DeletionResult:
    k: int
    abs_guided: float
    abs_random: float
    abs_anti: float
    gap: float
    gap_se: float
    flip_guided: float | None = None
    flip_random: float | None = None
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "k": self.k,
            "abs_guided": self.abs_guided,
            "abs_random": self.abs_random,
            "abs_anti": self.abs_anti,
            "gap": self.gap,
            "gap_se": self.gap_se,
        }
        if self.flip_guided is not None:
            d["flip_guided"] = self.flip_guided
            d["flip_random"] = self.flip_random
        d.update(self.extras)
        return d


def _mask(states: np.ndarray, idx: np.ndarray, reference: np.ndarray) -> np.ndarray:
    out = states.copy()
    rows = np.arange(len(states))[:, None]
    out[rows, idx] = reference[idx]
    return out


def deletion_test(
    head: nn.Module,
    states: np.ndarray,
    phi: np.ndarray,
    reference: np.ndarray,
    k: int,
    n_random: int = 20,
    seed: int = 123,
    track_sign_flips: bool = False,
) -> DeletionResult:
    """Guided vs random vs anti-guided top-k feature deletion on |target|.

    Guided deletes the k features with largest |phi|; anti the k smallest;
    random averages n_random draws. Displacement is |f(x_masked) - f(x)|
    per state; the gap is guided-minus-random with the SE of the paired
    per-state differences.
    """
    rng = np.random.default_rng(seed)
    dev = _device(head)
    with torch.no_grad():
        f0 = head(torch.from_numpy(states.astype(np.float32)).to(dev)).cpu().numpy()

    def displacement(masked: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            fm = head(torch.from_numpy(masked.astype(np.float32)).to(dev)).cpu().numpy()
        return fm

    order = np.argsort(-np.abs(phi), axis=1)
    guided_idx, anti_idx = order[:, :k], order[:, -k:]
    f_guided = displacement(_mask(states, guided_idx, reference))
    f_anti = displacement(_mask(states, anti_idx, reference))

    d = states.shape[1]
    rand_disp = np.zeros((n_random, len(states)))
    rand_flip = np.zeros(n_random)
    for r in range(n_random):
        ridx = np.stack([rng.choice(d, size=k, replace=False) for _ in range(len(states))])
        fr = displacement(_mask(states, ridx, reference))
        rand_disp[r] = np.abs(fr - f0)
        if track_sign_flips:
            rand_flip[r] = float((np.sign(fr) != np.sign(f0)).mean())

    disp_guided = np.abs(f_guided - f0)
    disp_rand = rand_disp.mean(axis=0)
    disp_anti = np.abs(f_anti - f0)
    paired = disp_guided - disp_rand
    res = DeletionResult(
        k=k,
        abs_guided=float(disp_guided.mean()),
        abs_random=float(disp_rand.mean()),
        abs_anti=float(disp_anti.mean()),
        gap=float(paired.mean()),
        gap_se=float(paired.std(ddof=1) / np.sqrt(len(paired))),
    )
    if track_sign_flips:
        res.flip_guided = float((np.sign(f_guided) != np.sign(f0)).mean())
        res.flip_random = float(rand_flip.mean())
    return res


# --------------------------------------------------------------------------
# Full dual-level study on one agent
# --------------------------------------------------------------------------


def dual_level_study(
    model,
    states: np.ndarray,
    feature_names: list[str],
    intervene_action: int = 1,
    ks: tuple[int, ...] = (1, 2),
    n_margin_min: int = 30,
    null_factor: float = 3.0,
    seed: int = 123,
    margin_pool: np.ndarray | None = None,
    max_margin_states: int = 500,
) -> dict:
    """Run both attributions under both tests (the paper's 2x2 cross matrix).

    Risk states: all provided states. Margin states: those where the policy's
    argmax action is ``intervene_action`` (the analog of the paper's explained
    intervention cases), scanned over ``margin_pool`` when given (policies that
    intervene rarely need the full prefix pool) and capped at
    ``max_margin_states``. Evaluability requires at least ``n_margin_min`` of
    them and a mean |margin| at least ``null_factor`` times the random
    displacement at the first k.
    """
    policy = model.policy
    critic, margin = CriticHead(policy), MarginHead(policy, intervene_action)
    wait_margin = WaitMarginHead(margin)
    intervene_logit = InterveneLogitHead(policy, intervene_action)
    wait_logit = WaitLogitHead(policy, intervene_action)
    reference = states.mean(axis=0)

    scan = states if margin_pool is None else margin_pool
    dev = _device(margin)
    with torch.no_grad():
        m0 = margin(torch.from_numpy(scan.astype(np.float32)).to(dev)).cpu().numpy()
    margin_states = scan[m0 > 0]
    if len(margin_states) > max_margin_states:
        idx = np.random.default_rng(seed).choice(
            len(margin_states), size=max_margin_states, replace=False
        )
        margin_states = margin_states[idx]

    wait_states = scan[m0 <= 0]
    if len(wait_states) > max_margin_states:
        idx = np.random.default_rng(seed).choice(
            len(wait_states), size=max_margin_states, replace=False
        )
        wait_states = wait_states[idx]

    out: dict = {
        "n_states": len(states),
        "n_margin_states": int(len(margin_states)),
        "n_wait_states": int(len(wait_states)),
        "feature_names": feature_names,
        "reference": reference.tolist(),
        "intervene_rate": float((m0 > 0).mean()),
    }

    phi_v = integrated_gradients(critic, states, reference)
    out["phi_v_mean_abs"] = np.abs(phi_v).mean(axis=0).tolist()
    out["value_test"] = {
        "phi_v": {
            str(k): deletion_test(critic, states, phi_v, reference, k, seed=seed).as_dict()
            for k in ks
        }
    }

    margin_ok = len(margin_states) >= n_margin_min
    out["margin_evaluable_sample"] = bool(margin_ok)
    if margin_ok:
        phi_dq = integrated_gradients(margin, margin_states, reference)
        out["phi_dq_mean_abs"] = np.abs(phi_dq).mean(axis=0).tolist()
        # target-above-null condition (Def. 4 (iii) analog)
        probe = deletion_test(
            margin, margin_states, phi_dq, reference, ks[0], seed=seed, track_sign_flips=True
        )
        with torch.no_grad():
            m_sel = margin(torch.from_numpy(margin_states.astype(np.float32)).to(dev)).cpu().numpy()
        mean_abs_margin = float(np.abs(m_sel).mean())
        out["mean_abs_margin"] = mean_abs_margin
        out["margin_above_null"] = bool(mean_abs_margin >= null_factor * probe.abs_random)
        out["margin_test"] = {
            "phi_dq": {
                str(k): deletion_test(
                    margin, margin_states, phi_dq, reference, k, seed=seed, track_sign_flips=True
                ).as_dict()
                for k in ks
            },
            # cross tests: each ranking on the other target
            "phi_v_on_margin": {
                str(k): deletion_test(
                    margin,
                    margin_states,
                    integrated_gradients(critic, margin_states, reference),
                    reference,
                    k,
                    seed=seed,
                    track_sign_flips=True,
                ).as_dict()
                for k in ks
            },
        }
        out["value_test"]["phi_dq_on_value"] = {
            str(k): deletion_test(critic, margin_states, phi_dq, reference, k, seed=seed).as_dict()
            for k in ks
        }

        # branch-wise attribution: phi(logit(intervene)) - phi(logit(wait)),
        # attribute-then-subtract (Padella et al. 2022 style), scored against
        # the same margin function as phi_dq above -- only the attribution
        # (which features get masked) differs, not what is measured.
        phi_intervene_branch = integrated_gradients(intervene_logit, margin_states, reference)
        phi_wait_branch = integrated_gradients(wait_logit, margin_states, reference)
        phi_dq_branchwise = phi_intervene_branch - phi_wait_branch
        out["phi_dq_branchwise_mean_abs"] = np.abs(phi_dq_branchwise).mean(axis=0).tolist()
        out["margin_test"]["phi_dq_branchwise"] = {
            str(k): deletion_test(
                margin,
                margin_states,
                phi_dq_branchwise,
                reference,
                k,
                seed=seed,
                track_sign_flips=True,
            ).as_dict()
            for k in ks
        }

        # multi-method before/after comparison (Saliency, DeepLift,
        # GradientShap, Occlusion, ShapleyValueSampling, KernelShap, Lime) --
        # see ATTRIBUTION_METHODS / _multi_method_comparison above.
        out["margin_test_by_method"] = _multi_method_comparison(
            margin, intervene_logit, wait_logit, margin_states, reference, ks, seed, branch_sign=1.0
        )

    wait_ok = len(wait_states) >= n_margin_min
    out["wait_margin_evaluable_sample"] = bool(wait_ok)
    if wait_ok:
        phi_dq_wait = integrated_gradients(wait_margin, wait_states, reference)
        out["phi_dq_wait_mean_abs"] = np.abs(phi_dq_wait).mean(axis=0).tolist()
        # target-above-null condition (Def. 4 (iii) analog), wait side
        probe_wait = deletion_test(
            wait_margin, wait_states, phi_dq_wait, reference, ks[0], seed=seed, track_sign_flips=True
        )
        with torch.no_grad():
            m_wait_sel = wait_margin(
                torch.from_numpy(wait_states.astype(np.float32)).to(dev)
            ).cpu().numpy()
        mean_abs_wait_margin = float(np.abs(m_wait_sel).mean())
        out["mean_abs_wait_margin"] = mean_abs_wait_margin
        out["wait_margin_above_null"] = bool(
            mean_abs_wait_margin >= null_factor * probe_wait.abs_random
        )
        out["wait_margin_test"] = {
            "phi_dq_wait": {
                str(k): deletion_test(
                    wait_margin, wait_states, phi_dq_wait, reference, k, seed=seed, track_sign_flips=True
                ).as_dict()
                for k in ks
            },
            "phi_v_on_wait_margin": {
                str(k): deletion_test(
                    wait_margin,
                    wait_states,
                    integrated_gradients(critic, wait_states, reference),
                    reference,
                    k,
                    seed=seed,
                    track_sign_flips=True,
                ).as_dict()
                for k in ks
            },
        }
        out["value_test"]["phi_dq_wait_on_value"] = {
            str(k): deletion_test(
                critic, wait_states, phi_dq_wait, reference, k, seed=seed
            ).as_dict()
            for k in ks
        }

        # branch-wise attribution, wait side: phi(logit(wait)) - phi(logit(intervene))
        # = -(phi(logit(intervene)) - phi(logit(wait))), attribute-then-subtract,
        # scored against wait_margin (same target as phi_dq_wait above).
        phi_intervene_branch_w = integrated_gradients(intervene_logit, wait_states, reference)
        phi_wait_branch_w = integrated_gradients(wait_logit, wait_states, reference)
        phi_dq_wait_branchwise = phi_wait_branch_w - phi_intervene_branch_w
        out["phi_dq_wait_branchwise_mean_abs"] = np.abs(phi_dq_wait_branchwise).mean(axis=0).tolist()
        out["wait_margin_test"]["phi_dq_wait_branchwise"] = {
            str(k): deletion_test(
                wait_margin,
                wait_states,
                phi_dq_wait_branchwise,
                reference,
                k,
                seed=seed,
                track_sign_flips=True,
            ).as_dict()
            for k in ks
        }

        # multi-method before/after comparison, wait side (branch = wait -
        # intervene, matching phi_dq_wait_branchwise's sign above).
        out["wait_margin_test_by_method"] = _multi_method_comparison(
            wait_margin, intervene_logit, wait_logit, wait_states, reference, ks, seed, branch_sign=-1.0
        )
    return out
