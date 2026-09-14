"""How the three RETIME levels compose -- the operators behind paper Section 6.

The policy's state is a function of the prefix and of two lower-level models::

    s_t = ( n(x_t),  g(r(x_t)),  p_T(x_t),  p_U(x_t) )

    n(x_t)      relative_position, available_resources   (policy-native)
    g(r)        (reliability, deviation), derived from the outcome predictor's
                r = P(deviant | x_t):  predicted = 1[r > 0.5], deviation = 1 -
                predicted, reliability = deviation (BPIC2012, ensemble
                agreement of one model) or the probability of the predicted
                class max(r, 1 - r) (BPIC2017)  -- risk_model.py's docstring
    p_T, p_U    the causal-effect estimator's two arms                (effect)

so the timing justification phi^{Delta Q} over the six coordinates of s_t can
be *propagated* to the prefix attributes through the risk explanation phi^r
and the two per-arm effect explanations phi^{p_T}, phi^{p_U}:

    phi^{Delta Q o x}_j = [j native] phi^{Delta Q}_j
                        + sum_{i in risk}   phi^{Delta Q}_i . phi^r_j     / sum_k phi^r_k
                        + sum_{i in effect} phi^{Delta Q}_i . phi^{m_i}_j / sum_k phi^{m_i}_k

the proportional ("rescale") propagation rule DeepSHAP uses between layers and
Chen, Lundberg & Lee (2022) formalise for a series of models. It conserves
completeness (the weights of each coordinate sum to one) and keeps the
*channel* through which each unit of the margin arrives (risk, effect-T,
effect-U, native), which is what lets a card say "CreditScore enters the
decision through the effect level by x and through the risk level by y".
When a lower-level attribution sums to (almost) nothing -- the model output
sits at its expected value -- the proportional weights are ill-defined and
the unit is split by |phi| instead (``FALLBACK_RATIO``); the number of such
splits is reported.

The rule is exact for a linear lower level and an approximation otherwise,
so :class:`EndToEndBox` exposes the composed function F(x, n) = Delta Q(s(x,
n)) -- prefix in, margin out, every model in between -- for a *direct*
attribution by Shapley value sampling (:func:`shapley_sampling`) and for the
paper's deletion test on the composition (:func:`deletion_test_e2e`): mask the
prefix attributes the propagated explanation ranks highest and see how far the
policy's margin moves, against the direct ranking, random and anti-guided
masking.

Two readings of the state coexist and both are supported (``rebuild_state``):
the *shipped* state (the pipeline's own numbers in the RL CSV, what the
checkpoint was trained on) and the *rebuilt* state (the same prefix scored by
the retrained risk and effect models, the numbers the lower levels actually
explain). The end-to-end function is by construction the rebuilt one.

Finally :func:`typology` cross-tabulates the decision points by (at risk?)
x (treatable?) x (acts?) and :func:`risk_effect_agreement` compares the two
lower-level explanations on the prefix vocabulary they share.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from dual_level import integrated_gradients

STATE_FEATS = ["relative_position", "reliability", "deviation", "available_resources", "Proba_if_Treated", "Proba_if_Untreated"]
NATIVE = ["relative_position", "available_resources"]
RISK = ["reliability", "deviation"]
EFFECT = ["Proba_if_Treated", "Proba_if_Untreated"]
LEVEL_OF = {**{f: "native" for f in NATIVE}, **{f: "risk" for f in RISK}, **{f: "effect" for f in EFFECT}}
CHANNELS = ("risk", "effect_T", "effect_U", "native")
FALLBACK_RATIO = 0.05  # |sum phi| below this fraction of sum |phi| -> split by |phi|


# ---------------------------------------------------------------------------
# The state as a function of the lower levels
# ---------------------------------------------------------------------------


def risk_features_from_r(log: str, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(reliability, deviation) the offline phase derives from r = P(deviant)."""
    r = np.asarray(r, dtype=np.float64)
    predicted = (r > 0.5).astype(np.float64)
    deviation = 1.0 - predicted
    if log == "BPIC2012":
        reliability = deviation.copy()
    else:
        reliability = np.maximum(r, 1.0 - r)
    return reliability, deviation


# The reward's counterfactual outcomes. p_T, p_U are P(undesired | treated /
# untreated) (effect_model.py: Y = 1 the deviant outcome), so the reward's
# y1 / y0 are the *desired* outcome, y = 1[p < 0.5], and the positive-effect
# rule y1 - y0 > 0 reads "the case ends well if treated and badly if not" --
# the convention SimBank's add_effect_features.py already used. Shoush &
# Dumas's released prepare_data_for_RL_V2.py thresholds p > 0.5 on the same
# P(deviant) columns (CausalLift treats Y = 1 as the conversion to raise),
# which pays the policy for intervening where the treatment is predicted to
# *cause* the undesired outcome; every coherent state here uses the corrected rule.
def desired_outcome(p: np.ndarray) -> np.ndarray:
    return (np.asarray(p, dtype=np.float64) < 0.5).astype(int)


def positive_effect_rule(pT: np.ndarray, pU: np.ndarray) -> np.ndarray:
    """y1 - y0 > 0: predicted to end well if treated and badly if untreated."""
    return (desired_outcome(pT) - desired_outcome(pU)) > 0


def build_state(log: str, rel: np.ndarray, r: np.ndarray, res: np.ndarray, pT: np.ndarray, pU: np.ndarray,
                feats: list[str] = STATE_FEATS) -> np.ndarray:
    """The state in the order of ``feats`` (the six-feature state by default;
    the released four-feature state when the effect features are absent)."""
    reliability, deviation = risk_features_from_r(log, r)
    cols = {"relative_position": np.asarray(rel, float), "reliability": reliability, "deviation": deviation,
            "available_resources": np.asarray(res, float), "Proba_if_Treated": np.asarray(pT, float), "Proba_if_Untreated": np.asarray(pU, float)}
    return np.stack([cols[f] for f in feats], axis=1).astype(np.float32)


def rebuild_state(log: str, states: np.ndarray, r: np.ndarray, pT: np.ndarray, pU: np.ndarray, feats: list[str] = STATE_FEATS) -> np.ndarray:
    """The shipped state with its lower-level coordinates replaced by the
    retrained models' outputs on the same prefix; the natives are kept.
    ``available_resources`` is optional (Sepsis's 5-feature state has no live
    capacity column, see pools.SEPSIS_FEATS): when absent from ``feats`` a
    dummy column is passed through build_state, which never reads it back
    since it is not in ``feats``."""
    res = states[:, feats.index("available_resources")] if "available_resources" in feats else np.zeros(len(states))
    return build_state(log, states[:, feats.index("relative_position")], r, res, pT, pU, feats)


def margin_of(policy, states: np.ndarray) -> np.ndarray:
    from dual_level import MarginHead

    head = MarginHead(policy, intervene_action=1)
    with torch.no_grad():
        return head(torch.from_numpy(np.asarray(states, np.float32))).numpy()


def timing_attribution(policy, states: np.ndarray, reference: np.ndarray, n_steps: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """phi of the side the policy chose on each state: IG of Delta Q where it
    acts, of Delta Q_wait = -Delta Q where it waits (IG is linear in the head,
    so the wait side is the negated act side). Returns (phi, sign)."""
    from dual_level import MarginHead

    head = MarginHead(policy, intervene_action=1)
    m = margin_of(policy, states)
    sign = np.where(m > 0, 1.0, -1.0)
    phi = integrated_gradients(head, np.asarray(states, np.float32), np.asarray(reference, np.float32), n_steps=n_steps)
    return phi * sign[:, None], sign


def level_shares(phi: np.ndarray, feats: list[str] = STATE_FEATS) -> dict:
    """Share of mean |phi| per feature and per level (Table 'card').

    Grouped Shapley importance over the risk/effect/native partition of the
    state, in the sense of Au et al., "Grouped Feature Importance and
    Combined Features Effect Plot" (Data Min. Knowl. Discov. 2022)."""
    g = np.abs(phi).mean(axis=0)
    share = g / g.sum() if g.sum() > 0 else np.zeros_like(g)
    per_feature = {f: float(s) for f, s in zip(feats, share)}
    per_level = {}
    for f, s in per_feature.items():
        per_level[LEVEL_OF[f]] = per_level.get(LEVEL_OF[f], 0.0) + s
    return {"per_feature": per_feature, "per_level": per_level}


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


def _weights(phi_lower: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise proportional weights phi_j / sum_k phi_k, with the |phi| split
    where the sum is too small to divide by. Returns (w, fallback_mask)."""
    s = phi_lower.sum(axis=1)
    a = np.abs(phi_lower).sum(axis=1)
    fallback = np.abs(s) < FALLBACK_RATIO * a
    w = np.zeros_like(phi_lower)
    ok = ~fallback & (a > 0)
    w[ok] = phi_lower[ok] / s[ok, None]
    fb = fallback & (a > 0)
    w[fb] = np.abs(phi_lower[fb]) / a[fb, None]
    return w, fallback


def propagate(phi_timing: np.ndarray, lower: dict[str, np.ndarray], feats: list[str] = STATE_FEATS) -> dict:
    """Push phi_timing (n x 6, over ``feats``) down to the prefix attributes.

    ``lower`` maps each non-native state feature to the attribution (n x D)
    of the lower-level model output it is derived from: reliability and
    deviation -> phi^r, Proba_if_Treated -> phi^{p_T}, Proba_if_Untreated ->
    phi^{p_U}. Returns the per-channel matrices (n x D), the native part
    (n x 2) and the composed vector over D + 2 inputs (prefix attributes
    first, then NATIVE), plus the number of fallback splits per feature.
    """
    n = phi_timing.shape[0]
    D = next(iter(lower.values())).shape[1]
    chan = {c: np.zeros((n, D)) for c in ("risk", "effect_T", "effect_U")}
    fallbacks = {}
    for i, f in enumerate(feats):
        if f in NATIVE:
            continue
        w, fb = _weights(lower[f])
        fallbacks[f] = int(fb.sum())
        c = "risk" if f in RISK else ("effect_T" if f == "Proba_if_Treated" else "effect_U")
        chan[c] += phi_timing[:, i:i + 1] * w
    # NATIVE members not in ``feats`` are simply absent (Sepsis's 5-feature
    # state has no available_resources, see pools.SEPSIS_FEATS): the native
    # block below has one column there, not always cp.NATIVE's two.
    native_feats = [f for f in NATIVE if f in feats]
    native = np.stack([phi_timing[:, feats.index(f)] for f in native_feats], axis=1)
    composed = np.concatenate([chan["risk"] + chan["effect_T"] + chan["effect_U"], native], axis=1)
    return {"channels": chan, "native": native, "composed": composed, "fallbacks": fallbacks}


def anchor_to_direct(phi_direct: np.ndarray, prop: dict) -> dict:
    """The per-decision variant of the composition: the attribution over the
    D + 2 inputs is the *direct* one of the composed function F (so its
    ranking is faithful to F by construction), and the propagation only says
    through which channel each unit arrives -- phi_direct_j is split among the
    channels in proportion to the propagated channel masses |ch_c[j]| (by
    |phi| among the risk/effect coordinates when the propagated channels carry
    nothing on that input). Returns the same structure as ``propagate``."""
    D = prop["channels"]["risk"].shape[1]
    masses = {c: np.abs(prop["channels"][c]) for c in ("risk", "effect_T", "effect_U")}
    tot = sum(masses.values())
    chan = {}
    for c, m in masses.items():
        w = np.where(tot > 0, m / np.where(tot > 0, tot, 1.0), 1.0 / 3)
        chan[c] = phi_direct[:, :D] * w
    native = phi_direct[:, D:]
    return {"channels": chan, "native": native, "composed": phi_direct.copy(), "fallbacks": {}}


def completeness_gap(phi_timing: np.ndarray, composed: np.ndarray) -> float:
    return float(np.abs(phi_timing.sum(axis=1) - composed.sum(axis=1)).max())


# ---------------------------------------------------------------------------
# The composed function F(x, n) = Delta Q(s(x, n))
# ---------------------------------------------------------------------------


class EndToEndBox:
    """Prefix attributes + native features in, the policy's margin out, every
    model of the pipeline in between (retrained risk and effect models, PPO).

    Inputs are a DataFrame ``Z`` with the prefix attributes (``raw_columns``,
    mixed types) followed by the two native columns; ``f(Z, sign)`` returns
    sign * Delta Q(s(Z)), so passing the sign of the decision on the unmasked
    state scores each state on the side the policy chose.
    """

    def __init__(self, log: str, risk_box, effect_box, policy, feats: list[str] = STATE_FEATS):
        self.log = log
        self.feats = list(feats)
        self.rbox, self.ebox, self.policy = risk_box, effect_box, policy
        if list(risk_box.columns) != list(effect_box.raw_columns):
            raise ValueError("risk and effect models must share the prefix vocabulary")
        self.raw_columns = list(risk_box.columns)
        self.cat_cols = set(risk_box.cat_cols)
        # native columns present in this log's state (Sepsis has no
        # available_resources, see pools.SEPSIS_FEATS -- one native column,
        # not cp.NATIVE's two).
        self.native = [f for f in NATIVE if f in feats]
        self.columns = self.raw_columns + self.native

    def inputs(self, X: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
        Z = X[self.raw_columns].copy().reset_index(drop=True)
        for f in self.native:
            Z[f] = states[:, self.feats.index(f)].astype(float)
        return Z

    def reference(self, Z: pd.DataFrame) -> pd.Series:
        return pd.Series({c: (Z[c].mode().iloc[0] if c in self.cat_cols else float(Z[c].astype(float).mean())) for c in self.columns})

    def lower(self, Z: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = Z[self.raw_columns]
        r = self.rbox.proba(X)
        pT, pU = self.ebox.probs(X)
        return r, pT, pU

    def state(self, Z: pd.DataFrame) -> np.ndarray:
        r, pT, pU = self.lower(Z)
        res = Z["available_resources"].to_numpy(float) if "available_resources" in self.native else np.zeros(len(Z))
        return build_state(self.log, Z["relative_position"].to_numpy(float), r, res, pT, pU, self.feats)

    def f(self, Z: pd.DataFrame, sign: np.ndarray | float = 1.0) -> np.ndarray:
        return np.asarray(sign, float) * margin_of(self.policy, self.state(Z))

    def mask(self, Z: pd.DataFrame, idx: np.ndarray, ref: pd.Series) -> pd.DataFrame:
        """Row i gets the columns idx[i] (indices into ``columns``) set to the reference."""
        out = Z.copy()
        for j, c in enumerate(self.columns):
            rows = np.where((idx == j).any(axis=1))[0]
            if len(rows):
                out.iloc[rows, out.columns.get_loc(c)] = ref[c]
        return out


def shapley_sampling(box: EndToEndBox, Z: pd.DataFrame, ref: pd.Series, sign: np.ndarray, n_perm: int = 20, seed: int = 123,
                     background: pd.DataFrame | None = None) -> np.ndarray:
    """Shapley values of f(Z, sign) by permutation sampling (Castro et al.).

    With ``background`` (rows in ``box.columns``), the interventional
    (pool-referenced) value: every permutation draws one background row per
    state and masks to it, so the mean over permutations telescopes to
    f(Z) - E_background f, the same reference the effect arms' TreeSHAP and,
    up to the Jensen gap, the timing level's mean state use. Without it, the
    single baseline ``ref`` (f(Z) - f(ref) exactly). Every permutation is
    shared across the rows and evaluated as one batch of D + 1 frames."""
    rng = np.random.default_rng(seed)
    n, D = len(Z), len(box.columns)
    phi = np.zeros((n, D))
    for _ in range(n_perm):
        perm = rng.permutation(D)
        if background is not None:
            B = background.iloc[rng.integers(0, len(background), size=n)].reset_index(drop=True)
        frames, cur = [Z.copy()], Z.copy()
        for j in perm:
            cur = cur.copy()
            col = box.columns[j]
            cur[col] = B[col].to_numpy() if background is not None else ref[col]
            frames.append(cur)
        vals = box.f(pd.concat(frames, ignore_index=True), np.tile(sign, D + 1)).reshape(D + 1, n)
        for step, j in enumerate(perm):
            phi[:, j] += vals[step] - vals[step + 1]
    return phi / n_perm


def deletion_test_e2e(box: EndToEndBox, Z: pd.DataFrame, ref: pd.Series, sign: np.ndarray, rankings: dict[str, np.ndarray],
                      k: int, n_random: int = 20, seed: int = 123) -> dict:
    """Guided deletion on the composed function for several rankings at once
    (each a |phi|-like n x D matrix), against random and anti-guided masking
    of the same k inputs; the gap of each ranking against random is the mean
    of the paired per-state differences with its SE."""
    rng = np.random.default_rng(seed)
    f0 = box.f(Z, sign)
    D = len(box.columns)
    rand = np.zeros((n_random, len(Z)))
    for r in range(n_random):
        ridx = np.stack([rng.choice(D, size=k, replace=False) for _ in range(len(Z))])
        rand[r] = np.abs(box.f(box.mask(Z, ridx, ref), sign) - f0)
    dr = rand.mean(axis=0)
    out = {"k": k, "abs_random": float(dr.mean()), "flip_random": float((np.sign(box.f(box.mask(Z, ridx, ref), sign)) != np.sign(f0)).mean())}
    for name, score in rankings.items():
        order = np.argsort(-np.abs(score), axis=1)
        fg = box.f(box.mask(Z, order[:, :k], ref), sign)
        fa = box.f(box.mask(Z, order[:, -k:], ref), sign)
        dg, da = np.abs(fg - f0), np.abs(fa - f0)
        paired = dg - dr
        se = float(paired.std(ddof=1) / np.sqrt(len(paired)))
        out[name] = {"abs_guided": float(dg.mean()), "abs_anti": float(da.mean()), "gap": float(paired.mean()), "gap_se": se,
                     "z": float(paired.mean() / se) if se > 0 else None, "flip_guided": float((np.sign(fg) != np.sign(f0)).mean())}
    return out


def ranking_agreement(a: np.ndarray, b: np.ndarray, k: int = 5) -> dict:
    """Per-state Spearman of |a| vs |b| (mean, median) and Jaccard of the
    top-k sets, plus the global Spearman of the mean-|phi| rankings.

    Rank correlation and top-k feature agreement in the sense of Krishna
    et al., "The Disagreement Problem in Explainable Machine Learning"
    (TMLR 2022)."""
    from scipy.stats import spearmanr

    rhos, jacc = [], []
    for x, y in zip(np.abs(a), np.abs(b)):
        if x.std() > 0 and y.std() > 0:
            rhos.append(spearmanr(x, y)[0])
        tx, ty = set(np.argsort(-x)[:k]), set(np.argsort(-y)[:k])
        jacc.append(len(tx & ty) / len(tx | ty))
    ga, gb = np.abs(a).mean(axis=0), np.abs(b).mean(axis=0)
    return {"spearman_mean": float(np.mean(rhos)), "spearman_median": float(np.median(rhos)),
            f"jaccard_top{k}_mean": float(np.mean(jacc)), "global_spearman": float(spearmanr(ga, gb)[0]),
            "top1_agreement": float((np.argmax(np.abs(a), axis=1) == np.argmax(np.abs(b), axis=1)).mean())}


# ---------------------------------------------------------------------------
# Relations between the levels
# ---------------------------------------------------------------------------


def typology(risky: np.ndarray, treatable: np.ndarray, acts: np.ndarray, phi_timing: np.ndarray | None = None) -> dict:
    """Cross-tabulation (at risk?) x (treatable?) x (acts?) with counts and,
    when phi_timing is given, the level shares inside each cell."""
    cells = {}
    for rk in (True, False):
        for tr in (True, False):
            for ac in (True, False):
                m = (risky == rk) & (treatable == tr) & (acts == ac)
                key = f"{'risky' if rk else 'safe'}|{'treatable' if tr else 'untreatable'}|{'act' if ac else 'wait'}"
                cells[key] = {"n": int(m.sum())}
                if phi_timing is not None and m.sum() > 0:
                    cells[key]["level_share"] = level_shares(phi_timing[m])["per_level"]
    return cells


def risk_effect_agreement(phi_r: np.ndarray, phi_cate: np.ndarray, names: list[str], k: int = 10) -> dict:
    """How the risk and effect explanations relate on the shared prefix
    vocabulary: global rank agreement, top-k overlap, and per-attribute sign
    agreement (does the attribute push toward a bad outcome and toward a
    larger effect at the same time?) on the states where both are non-zero.

    Rank agreement, top-k feature agreement and sign agreement in the sense
    of Krishna et al., "The Disagreement Problem in Explainable Machine
    Learning" (TMLR 2022)."""
    from scipy.stats import spearmanr

    gr, ge = np.abs(phi_r).mean(axis=0), np.abs(phi_cate).mean(axis=0)
    top_r, top_e = set(np.argsort(-gr)[:k]), set(np.argsort(-ge)[:k])
    both = (phi_r != 0) & (phi_cate != 0)
    same = (np.sign(phi_r) == np.sign(phi_cate)) & both
    per_attr = {}
    for j in sorted(top_r | top_e, key=lambda j: -(gr[j] / gr.sum() + ge[j] / ge.sum())):
        nb = int(both[:, j].sum())
        per_attr[names[j]] = {"risk_share": float(gr[j] / gr.sum()), "effect_share": float(ge[j] / ge.sum()),
                              "sign_agreement": float(same[:, j].sum() / nb) if nb else None, "n_both": nb}
    return {"global_spearman": float(spearmanr(gr, ge)[0]), f"jaccard_top{k}": len(top_r & top_e) / len(top_r | top_e),
            "shared_top": [names[j] for j in top_r & top_e], "per_attribute": per_attr,
            "sign_agreement_overall": float(same.sum() / both.sum()) if both.sum() else None}


# ---------------------------------------------------------------------------
# Cancellation between channels, well-definedness, baseline alignment
# ---------------------------------------------------------------------------


def cancellation(prop: dict) -> dict:
    """How much of the channel mass cancels inside an input.

    For state s and prefix attribute j the channels contribute
    ch_c[s, j] (c in risk, effect_T, effect_U) and the composed value is
    their sum; the cancellation index is 1 - |sum_c ch_c| / sum_c |ch_c|,
    zero when the channels agree in sign and one when they cancel exactly.
    Returns the per-state x per-attribute matrix, the mass-weighted index
    per attribute (over states) and per state (over attributes), and the
    global one.

    This is the failure mode DeepLIFT's RevealCancel rule was introduced to
    catch: the plain Rescale rule of `propagate` (DeepSHAP's rule, Shrikumar
    et al., "Learning Important Features Through Propagating Activation
    Differences", ICML 2017) does not separate positive and negative
    contributions to a shared downstream unit, so opposite-sign channels can
    net out even though each carried real mass; the index reports how much
    of that mass Rescale hides.
    """
    chans = [prop["channels"][c] for c in ("risk", "effect_T", "effect_U")]
    net = np.abs(sum(chans))
    mass = sum(np.abs(c) for c in chans)
    with np.errstate(invalid="ignore", divide="ignore"):
        per = np.where(mass > 0, 1.0 - net / mass, 0.0)
    per_attr = 1.0 - net.sum(axis=0) / np.where(mass.sum(axis=0) > 0, mass.sum(axis=0), 1.0)
    per_state = 1.0 - net.sum(axis=1) / np.where(mass.sum(axis=1) > 0, mass.sum(axis=1), 1.0)
    return {"matrix": per, "per_attribute": per_attr, "per_state": per_state, "overall": float(1.0 - net.sum() / mass.sum()) if mass.sum() > 0 else 0.0}


def well_defined(lower: dict[str, np.ndarray], ratio: float = FALLBACK_RATIO) -> dict:
    """The non-degeneracy condition of the composition, per coordinate: the
    lower-level attribution must sum to at least ``ratio`` of its |phi| mass
    for the proportional weights to be defined. Reports the share of states
    that satisfy it, the median ratio, and the share of states where all
    coordinates do."""
    out, ok_all = {}, None
    for f, phi in lower.items():
        s, a = phi.sum(axis=1), np.abs(phi).sum(axis=1)
        r = np.where(a > 0, np.abs(s) / np.where(a > 0, a, 1.0), 0.0)
        ok = r >= ratio
        out[f] = {"share_defined": float(ok.mean()), "median_ratio": float(np.median(r)), "n_fallback": int((~ok).sum())}
        ok_all = ok if ok_all is None else (ok_all & ok)
    out["ratio"] = ratio
    out["share_all_defined"] = float(ok_all.mean())
    return out


def baseline_alignment(phi_lower: np.ndarray, logit_output: np.ndarray) -> dict:
    """How far a lower-level attribution's implicit baseline is from the pool
    reference the timing level uses. ``phi_lower`` sums to logit m(x) minus
    the explainer's expected value; the pool-aligned sum is logit m(x) minus
    the pool mean logit. Reports the gap between the two baselines, the sign
    agreement of the two sums (does the attribution push the coordinate in the
    direction of its distance from the pool mean?) and their correlation."""
    s_tree = phi_lower.sum(axis=1)
    s_pool = logit_output - logit_output.mean()
    ev_tree = float((logit_output - s_tree).mean())
    both = (np.abs(s_tree) > 1e-9) & (np.abs(s_pool) > 1e-9)
    return {"expected_value": ev_tree, "pool_mean_logit": float(logit_output.mean()), "baseline_gap": float(logit_output.mean() - ev_tree),
            "sign_agreement": float((np.sign(s_tree[both]) == np.sign(s_pool[both])).mean()) if both.any() else None,
            "corr": float(np.corrcoef(s_tree, s_pool)[0, 1]) if s_tree.std() > 0 else None}


def per_state_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.stats import spearmanr

    out = np.full(len(a), np.nan)
    for i, (x, y) in enumerate(zip(np.abs(a), np.abs(b))):
        if x.std() > 0 and y.std() > 0:
            out[i] = spearmanr(x, y)[0]
    return out


def sign_groups(per_attribute: dict, agree: float = 0.8, oppose: float = 0.2) -> dict:
    """Classify the shared-vocabulary attributes by whether the risk and effect
    explanations push them the same way (both risky and treatable), opposite
    ways (risky but not treatable, or the reverse) or neither."""
    groups = {"agree": [], "oppose": [], "independent": [], "one_level_only": []}
    for name, d in per_attribute.items():
        s = d.get("sign_agreement")
        if s is None:
            groups["one_level_only"].append(name)
        elif s >= agree:
            groups["agree"].append(name)
        elif s <= oppose:
            groups["oppose"].append(name)
        else:
            groups["independent"].append(name)
    return groups
