"""Explanation-quality metrics of the PPM-explainability literature, ported
to a policy over a tabular state.

Each metric keeps the formula of the code base it comes from; where that
formula needs a label or a training set the policy does not have, the
substitution is stated in the docstring. Feature *families* replace the
event / case / control-flow attribute families the reference metrics are
defined over (see ``FAMILIES``).

* ``parsimony``                -- Stevens & De Smedt ``metrics/parsimony.py``;
                                  Warmuth & Leopold ``experiment_evaluation.py``
* ``functional_complexity``    -- Stevens & De Smedt
                                  ``metrics/functional_complexity.py`` (L1 form)
                                  and Stevens et al. (flip-rate form)
* ``monotonicity``             -- Stevens & De Smedt ``metrics/faithfulness.py``
                                  (Spearman) and Stevens et al. (Kendall)
* ``lod``                      -- Stevens & De Smedt ``metrics/faithfulness.py``
* ``consisxai``                -- Elkhawaga et al., ConsisXAI:
                                  ``Consistency_ratios.py``,
                                  ``Consistency_measures.py``
* ``agreement``                -- Elkhawaga et al., cross-model / cross-run
                                  top-k comparisons, made numeric
* ``rediscovery_rate``         -- Warmuth & Leopold ``utils.calculate_overlap``
                                  with identity matching (no word2vec)
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import kendalltau, spearmanr

# The three attribute families of this policy's state, standing in for the
# event / case / control-flow families of the reference metrics.
FAMILIES: dict[str, list[str]] = {
    "progress": ["relative_position"],
    "prediction": ["reliability", "deviation"],
    "capacity": ["available_resources"],
    # only present in the "cate" policy variant (pools.VARIANTS)
    "effect": ["Proba_if_Treated", "Proba_if_Untreated"],
}


def family_of(feature: str) -> str:
    for fam, members in FAMILIES.items():
        if feature in members:
            return fam
    raise KeyError(feature)


# ---------------------------------------------------------------------------
# Parsimony
# ---------------------------------------------------------------------------


def parsimony(global_importance: np.ndarray, feature_names: list[str], share_threshold: float = 0.05, eps: float = 1e-9) -> dict:
    """Number of features an explanation actually uses.

    * ``nonzero``: |importance| > eps -- the reference count (features with
      a non-zero coefficient / importance). A dense network gives every
      feature a non-zero gradient, so this is 4 by construction here.
    * ``above_share``: share of the total |importance| at least
      ``share_threshold`` -- the tolerance Warmuth & Leopold apply
      (``round(importance, 1) >= 0.005``) generalised to a relative one.
    * ``effective``: exp(entropy of the shares), the number of equally
      weighted features that would give the same entropy (reported as a
      continuous companion; not in the reference code).
    * per-family counts of the ``above_share`` set.
    """
    imp = np.abs(np.asarray(global_importance, dtype=np.float64))
    total = imp.sum()
    share = imp / total if total > 0 else np.zeros_like(imp)
    nonzero = imp > eps
    above = share >= share_threshold
    p = share[share > 0]
    effective = float(np.exp(-(p * np.log(p)).sum())) if len(p) else 0.0
    by_family = {fam: int(sum(above[i] for i, f in enumerate(feature_names) if f in members)) for fam, members in FAMILIES.items()}
    return {
        "nonzero": int(nonzero.sum()),
        "above_share": int(above.sum()),
        "share_threshold": share_threshold,
        "effective": effective,
        "shares": share.tolist(),
        "by_family": by_family,
    }


# ---------------------------------------------------------------------------
# Functional complexity
# ---------------------------------------------------------------------------


def functional_complexity(f, states: np.ndarray, feature_names: list[str], seed: int = 42) -> dict:
    """How much the policy's output changes when a feature (or a whole
    family) is replaced by another observed value, per state, averaged.

    * ``l1``: Stevens & De Smedt, ``100 * mean_i |f(x_i) - f(x'_i)|`` where x'
      replaces every column of the family by another observed value of that
      column. With a hard-label ``f`` this is 100x the flip rate; on the
      continuous target it is the mean absolute displacement x100.
    * ``flip``: Stevens et al.'s ``NF / n_instances`` -- the fraction of
      states whose argmax action changes (sign of the margin flips). Their
      notebook mutates the frame cumulatively across features; here each
      feature is perturbed from the clean state, per family and per
      feature, which is the intent the paper states.
    """
    from xai_methods import replace_with_other_observed

    rng = np.random.default_rng(seed)
    f0 = f(states)
    idx = {name: j for j, name in enumerate(feature_names)}

    def one(cols: list[int]) -> dict:
        pert = states.copy()
        for j in cols:
            pert[:, j] = replace_with_other_observed(states[:, j], rng)
        fp = f(pert)
        return {"l1": float(100.0 * np.mean(np.abs(fp - f0))), "flip": float(np.mean(np.sign(fp) != np.sign(f0)))}

    per_feature = {name: one([idx[name]]) for name in feature_names}
    per_family = {fam: one([idx[m] for m in members if m in idx]) for fam, members in FAMILIES.items()}
    total = one(list(range(states.shape[1])))
    return {"per_feature": per_feature, "per_family": per_family, "all_features": total}


# ---------------------------------------------------------------------------
# Monotonicity (explanation vs. model effect)
# ---------------------------------------------------------------------------


def monotonicity(global_importance: np.ndarray, model_effects: np.ndarray) -> dict:
    """Rank agreement between the explanation's global importance and the
    model's own per-feature effect (``xai_methods.var_importance``).
    Spearman (Stevens & De Smedt, Warmuth & Leopold) and Kendall (Stevens
    et al.); with four features Spearman takes only a handful of distinct
    values, so both are reported with their p-values."""
    a = np.abs(np.asarray(global_importance, dtype=np.float64))
    b = np.asarray(model_effects, dtype=np.float64)
    rho, p_rho = spearmanr(a, b)
    tau, p_tau = kendalltau(a, b)
    return {"spearman": float(rho), "spearman_p": float(p_rho), "kendall": float(tau), "kendall_p": float(p_tau)}


# ---------------------------------------------------------------------------
# LOD -- level of disagreement over attribute families
# ---------------------------------------------------------------------------


def _family_counts(names_sorted: list[str], k: int) -> list[int]:
    top = names_sorted[:k]
    return [sum(1 for n in top if n in members) for members in FAMILIES.values()]


def lod(global_importance: np.ndarray, model_effects: np.ndarray, feature_names: list[str], k: int = 2) -> dict:
    """Stevens & De Smedt's LOD: Euclidean distance between the family
    composition of the explanation's top-k features and of the model's
    top-k features by permutation effect. Their k is 10 over ~100 columns;
    with four features k=2 is the analogous half-of-the-features cut. A
    continuous companion, ``normalised_distance``, is the Euclidean distance
    between the two importance vectors after each is scaled to sum to one."""
    a = np.abs(np.asarray(global_importance, dtype=np.float64))
    b = np.asarray(model_effects, dtype=np.float64)
    names_a = [feature_names[i] for i in np.argsort(-a, kind="stable")]
    names_b = [feature_names[i] for i in np.argsort(-b, kind="stable")]
    ca, cb = _family_counts(names_a, k), _family_counts(names_b, k)
    an = a / a.sum() if a.sum() > 0 else a
    bn = b / b.sum() if b.sum() > 0 else b
    return {
        "k": k,
        "families": list(FAMILIES),
        "explanation_counts": ca,
        "model_counts": cb,
        "lod": float(np.linalg.norm(np.asarray(ca) - np.asarray(cb))),
        "normalised_distance": float(np.linalg.norm(an - bn)),
    }


# ---------------------------------------------------------------------------
# ConsisXAI: reducts, core, consistency ratio, AIC / BIC
# ---------------------------------------------------------------------------


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _gini(p: np.ndarray) -> float:
    return float(1.0 - (p ** 2).sum())


def _discretise(col: np.ndarray, n_bins: int = 10) -> np.ndarray:
    uniq = np.unique(col)
    if len(uniq) <= n_bins:
        return np.searchsorted(uniq, col)
    edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))
    return np.clip(np.searchsorted(edges, col, side="right") - 1, 0, len(edges) - 2)


def _information_gain(col: np.ndarray, y: np.ndarray, impurity) -> float:
    """ConsisXAI's ``comp_feature_information_gain``: H(y) - sum_levels
    w_level H(y | level), with H = entropy or gini."""
    levels = _discretise(col)
    classes = np.unique(y)

    def dist(mask):
        return np.array([(y[mask] == c).mean() for c in classes])

    base = impurity(dist(np.ones(len(y), bool)))
    cond = 0.0
    for lv in np.unique(levels):
        m = levels == lv
        cond += m.mean() * impurity(dist(m))
    return float(base - cond)


def _information_value(col: np.ndarray, y: np.ndarray) -> float:
    """Weight-of-evidence information value, as ConsisXAI's
    ``calculate_woe_iv`` (bins = the discretised feature), with the usual
    +0.5 count adjustment so that a bin holding only one class (a feature
    that separates the target perfectly, which happens here) yields a large
    finite IV instead of being dropped."""
    levels = _discretise(col)
    good, bad = (y == 1), (y == 0)
    n_good, n_bad = max(good.sum(), 1), max(bad.sum(), 1)
    iv = 0.0
    for lv in np.unique(levels):
        m = levels == lv
        dg = ((m & good).sum() + 0.5) / (n_good + 0.5)
        db = ((m & bad).sum() + 0.5) / (n_bad + 0.5)
        iv += (dg - db) * math.log(dg / db)
    return float(iv)


def criteria_scores(f, states: np.ndarray, feature_names: list[str], model_effects: np.ndarray, seed: int = 42) -> dict:
    """ConsisXAI's nine per-feature scoring criteria, for a policy.

    Their four *embedded* criteria (logit / XGB / RF / GBM importances of
    models trained on the data) become model-derived effects of the policy
    itself: permutation effect (``model_effects``), occlusion to the pool
    mean, mean |gradient| and per-feature functional complexity. Their five
    *filter* criteria (information gain with entropy and gini, information
    value, chi2/ANOVA F, TuRF) are computed against the binary target
    "wait margin above its pool median" -- the policy's own decision
    strength, in place of the class label -- with ``mutual_info_classif``
    standing in for TuRF (skrebate is not available). Returns the raw
    scores; ``reducts_and_core`` scales them exactly as the reference does.
    """
    from sklearn.feature_selection import f_classif, mutual_info_classif

    from xai_methods import replace_with_other_observed

    rng = np.random.default_rng(seed)
    f0 = f(states)
    y = (f0 > np.median(f0)).astype(int)
    D = states.shape[1]
    ref = states.mean(axis=0)

    occl = np.zeros(D)
    fc = np.zeros(D)
    for j in range(D):
        m = states.copy()
        m[:, j] = ref[j]
        occl[j] = float(np.mean(np.abs(f(m) - f0)))
        p = states.copy()
        p[:, j] = replace_with_other_observed(states[:, j], rng)
        fc[j] = float(np.mean(np.abs(f(p) - f0)))

    # mean |gradient| via finite differences of the black box (no autograd needed)
    grad = np.zeros(D)
    h = 1e-3
    for j in range(D):
        up = states.copy()
        up[:, j] += h
        grad[j] = float(np.mean(np.abs((f(up) - f0) / h)))

    ig_ent = np.array([_information_gain(states[:, j], y, _entropy) for j in range(D)])
    ig_gini = np.array([_information_gain(states[:, j], y, _gini) for j in range(D)])
    iv = np.array([_information_value(states[:, j], y) for j in range(D)])
    if y.min() == y.max():
        anova = np.zeros(D)
        mi = np.zeros(D)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # constant columns give 0/0 in f_classif
            anova = np.nan_to_num(f_classif(states, y)[0])
        mi = mutual_info_classif(states, y, random_state=seed)

    return {
        "target": "target above its pool median (binary)",
        "criteria": {
            "embedded_permutation": np.asarray(model_effects, dtype=np.float64).tolist(),
            "embedded_occlusion": occl.tolist(),
            "embedded_gradient": grad.tolist(),
            "embedded_functional_complexity": fc.tolist(),
            "information_gain_entropy": ig_ent.tolist(),
            "information_gain_gini": ig_gini.tolist(),
            "information_value": iv.tolist(),
            "anova_f": np.asarray(anova, dtype=np.float64).tolist(),
            "mutual_information": np.asarray(mi, dtype=np.float64).tolist(),
        },
        "feature_names": list(feature_names),
    }


def _minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    lo, hi = v.min(), v.max()
    if hi - lo <= 0:
        return np.ones_like(v)
    return (v - lo) / (hi - lo)


def reducts_and_core(criteria: dict, feature_names: list[str], seed: int = 42, max_attempts: int = 100) -> dict:
    """ConsisXAI's reducts and core.

    Each criterion column is shifted to be non-negative and MinMax-scaled to
    [0, 1]; the threshold grid runs from the smallest to the largest column
    mean in 0.01 steps. Starting at the minimum, a reduct per criterion is
    the set of features scoring at least the threshold, the core is the
    intersection of all reducts; while the core is empty a new threshold
    is drawn at random from the grid (``get_threshold``), up to
    ``max_attempts``. Core weights are the feature's mean score over the
    reducts. The *selected* reduct is the shortest one that is not the
    whole feature set (``main.py``), or the shortest one if every reduct is
    the whole set.
    """
    rng = np.random.default_rng(seed)
    names = list(criteria)
    M = np.column_stack([np.asarray(criteria[n], dtype=np.float64) for n in names])
    M = M - np.minimum(M.min(axis=0), 0.0)  # shift negative columns up
    S = np.column_stack([_minmax(M[:, c]) for c in range(M.shape[1])])
    col_means = S.mean(axis=0)
    t_min, t_max = float(col_means.min()), float(col_means.max())
    grid = np.round(np.arange(t_min, t_max + 1e-9, 0.01), 2) if t_max > t_min else np.array([t_min])

    def compute(threshold):
        reds = {n: [feature_names[i] for i in range(len(feature_names)) if S[i, c] >= threshold] for c, n in enumerate(names)}
        weights = {n: {feature_names[i]: float(S[i, c]) for i in range(len(feature_names)) if S[i, c] >= threshold} for c, n in enumerate(names)}
        core = sorted(set.intersection(*(set(v) for v in reds.values()))) if reds else []
        return reds, weights, core

    threshold = t_min
    reds, weights, core = compute(threshold)
    attempts = 0
    while not core and attempts < max_attempts:
        threshold = float(rng.choice(grid))
        reds, weights, core = compute(threshold)
        attempts += 1

    core_weights = {fe: float(np.mean([w[fe] for w in weights.values() if fe in w])) for fe in core}
    lengths = {n: len(r) for n, r in reds.items()}
    candidates = [n for n, r in reds.items() if len(r) != len(feature_names) and len(r) > 0]
    if candidates:
        selected = min(candidates, key=lambda n: lengths[n])
    else:
        nonempty = [n for n, r in reds.items() if len(r) > 0]
        selected = min(nonempty, key=lambda n: lengths[n]) if nonempty else names[0]
    return {
        "scaled_scores": {n: S[:, c].tolist() for c, n in enumerate(names)},
        "threshold": threshold,
        "threshold_min": t_min,
        "threshold_max": t_max,
        "attempts": attempts,
        "reducts": reds,
        "core": core,
        "core_weights": core_weights,
        "selected_reduct_criterion": selected,
        "selected_reduct": reds[selected],
    }


def xai_top_k(global_importance: np.ndarray, feature_names: list[str], k: int) -> dict:
    """The explanation's top-k features with their importance MinMax-scaled
    across those k (ConsisXAI's ``retrieve_vector`` + per-key scaling). A
    single feature (k=1) scales to 1, where sklearn's scaler would give 0
    and make the score-weighted ratio vanish by construction."""
    imp = np.abs(np.asarray(global_importance, dtype=np.float64))
    order = np.argsort(-imp, kind="stable")[:k]
    scaled = _minmax(imp[order])
    return {feature_names[i]: float(s) for i, s in zip(order, scaled)}


def consistency_ratio(selected: dict, reference: list[str]) -> dict:
    """ConsisXAI's ``ComputeRatio``: regular = sum of the scaled scores of
    the features in both the explanation's top-k and the reference subset,
    over |reference|; experimental = |intersection| / |reference|."""
    inter = [fe for fe in selected if fe in reference]
    n_ref = max(len(reference), 1)
    score_sum = float(sum(selected[fe] for fe in inter))
    return {
        "regular": score_sum / n_ref,
        "experimental": len(inter) / n_ref,
        "intersection": inter,
        "intersection_size": len(inter),
        "intersection_score_sum": score_sum,
        "missed": len(reference) - len(inter),
    }


def abic(n: int, ratio: float, num_params: int) -> dict:
    """ConsisXAI's ``calculate_abic``: with the complement 1 - ratio as the
    likelihood term, AIC = -2 log2(1 - ratio) + 2 p and BIC = -2 log2(1 -
    ratio) + p log2(n), p = number of reference features the explanation
    missed. A ratio of exactly 1 leaves the log undefined (None)."""
    comp = 1.0 - ratio
    if comp <= 0:
        return {"aic": None, "bic": None}
    ll = -2.0 * math.log2(comp)
    return {"aic": ll + 2 * num_params, "bic": ll + num_params * math.log2(max(n, 2))}


def consisxai(global_importance: np.ndarray, feature_names: list[str], rc: dict, n: int) -> dict:
    """The full ConsisXAI evaluation of one explanation against the
    reducts/core in ``rc`` (from ``reducts_and_core``)."""
    out = {}
    for label, reference in (("reduct", rc["selected_reduct"]), ("core", rc["core"])):
        k = max(len(reference), 1)
        sel = xai_top_k(global_importance, feature_names, k)
        ratio = consistency_ratio(sel, reference)
        entry = {"reference": reference, "selected": sel, **ratio}
        for variant in ("regular", "experimental"):
            entry[f"abic_{variant}"] = abic(n, ratio[variant], ratio["missed"])
        out[label] = entry
    return out


# ---------------------------------------------------------------------------
# Agreement between explanations (cross-method, cross-pool, cross-log)
# ---------------------------------------------------------------------------


def agreement(importances: dict[str, np.ndarray], feature_names: list[str], k: int = 2) -> dict:
    """Pairwise agreement of global importance vectors: Spearman, Kendall,
    top-1 identity and Jaccard of the top-k sets (the numeric form of
    Elkhawaga et al.'s side-by-side top-k tables and dependence overlays)."""
    names = list(importances)
    vecs = {n: np.abs(np.asarray(v, dtype=np.float64)) for n, v in importances.items()}
    tops = {n: [feature_names[i] for i in np.argsort(-vecs[n], kind="stable")] for n in names}
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho = spearmanr(vecs[a], vecs[b])[0]
            tau = kendalltau(vecs[a], vecs[b])[0]
            sa, sb = set(tops[a][:k]), set(tops[b][:k])
            pairs[f"{a}|{b}"] = {
                "spearman": float(rho) if np.isfinite(rho) else None,
                "kendall": float(tau) if np.isfinite(tau) else None,
                "top1_same": tops[a][0] == tops[b][0],
                f"jaccard_top{k}": len(sa & sb) / len(sa | sb),
            }
    return {"ranking": {n: tops[n] for n in names}, "pairs": pairs}


# ---------------------------------------------------------------------------
# Rediscovery rate (Warmuth & Leopold), identity matching
# ---------------------------------------------------------------------------


def rediscovery_rate(global_importance: np.ndarray, feature_names: list[str], ground_truth: list[str], share_threshold: float = 0.05) -> dict:
    """Fraction of the ground-truth features that appear among the features
    the explanation gives at least ``share_threshold`` of its weight
    (``calculate_overlap`` with the word2vec similarity replaced by identity,
    since state features are not words). Only meaningful when the features
    that truly drive the decision are known, e.g. on a synthetic policy."""
    imp = np.abs(np.asarray(global_importance, dtype=np.float64))
    share = imp / imp.sum() if imp.sum() > 0 else imp
    found = {feature_names[i] for i in range(len(feature_names)) if share[i] >= share_threshold}
    matches = [g for g in ground_truth if g in found]
    return {"rate": len(matches) / max(len(ground_truth), 1), "matches": matches}
