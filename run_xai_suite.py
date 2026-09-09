"""The explanation methods and explanation-quality metrics of the PPM
explainability literature, run on this repository's three PPO checkpoints.

What the five reference code bases produce, and where it is reproduced here:

  Warmuth & Leopold 2022 (explainable-predictive-process-monitoring-with-text)
      SHAP (shap.Explainer) bar / beeswarm / waterfall / force; parsimony;
      monotonicity (Spearman vs. the model's own importance); rediscovery rate
      -> xai_methods.kernel_shap/deep_shap, xai_plots.shap_*,
         xai_metrics.parsimony / monotonicity / rediscovery_rate
  Elkhawaga et al. 2022 (PPM_XAI_Comparison)
      SHAP global + local (force, decision, dependence), LIME with CSI/VSI
      stability, permutation importance (10 repeats, box plots), ALE, logit
      coefficients, correlation heat maps, mutual information, cross-model /
      cross-run top-k comparisons, execution times, KMeans-selected local
      instances (nearest / farthest to each centroid)
      -> xai_methods.lime_*, permutation_importance, ale_all;
         xai_plots.*; xai_metrics.agreement; ``local_instances`` below
  Stevens et al. (Quantifying-Explainability)
      parsimony, functional complexity (flip rate), monotonicity (Spearman
      and Kendall)
      -> xai_metrics.parsimony / functional_complexity / monotonicity
  Elkhawaga et al. 2023 (ConsisXAI)
      reducts and core from nine scoring criteria, reduct / core consistency
      ratio (score-weighted and count-based), AIC / BIC
      -> xai_metrics.criteria_scores / reducts_and_core / consisxai
  Stevens & De Smedt (Explainability-in-Process-Outcome-Prediction)
      parsimony per attribute family, functional complexity (L1) per family,
      monotonicity vs. observed-value permutation effects, LOD
      -> xai_methods.var_importance, xai_metrics.* with FAMILIES

Every method explains the same target: by default the paper's wait-side
margin Delta Q_wait(s) = log pi(wait|s) - log pi(intervene|s) on the wait
states of the Section 6/7 evaluation pool (one prefix per case, 500 cases,
seed 123; pools.py). ``--target`` switches to the intervene margin, an action
probability or the critic. The paper's own Integrated Gradients attribution
is scored with the same metrics, and every method's ranking is also put
through the paper's deletion test (dual_level.deletion_test), so the two
evaluation traditions can be read side by side.

Usage:
    python run_xai_suite.py                       # all three logs, all methods
    python run_xai_suite.py --logs BPIC2017 --quick
Writes paths.XAI_JSON and paths.XAI_FIGURES/<log>/*.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

import paths
import pools
import xai_metrics as xm
import xai_methods as xme
import xai_plots as xp
from dual_level import deletion_test

LOGS = list(pools.LOGS)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def local_instances(states: np.ndarray, seed: int = 123) -> dict:
    """Elkhawaga et al.'s local-explanation protocol: standardise the pool,
    KMeans with the number of clusters (2..6) of best silhouette, then the
    state nearest to and the state farthest from each centroid."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(states.astype(np.float64))
    best = None
    for k in range(2, 7):
        if k >= len(Z):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        s = silhouette_score(Z, km.labels_)
        if best is None or s > best[0]:
            best = (s, k, km)
    s, k, km = best
    picks = {}
    for c in range(k):
        members = np.where(km.labels_ == c)[0]
        d = np.linalg.norm(Z[members] - km.cluster_centers_[c], axis=1)
        picks[f"cluster{c}_nearest"] = int(members[np.argmin(d)])
        picks[f"cluster{c}_farthest"] = int(members[np.argmax(d)])
    return {"n_clusters": k, "silhouette": float(s), "picks": picks}


def run_log(name: str, model_path, states_pool: np.ndarray, args, feats: list[str]) -> dict:
    FEATS = list(feats)  # the state's feature names for this log (4, or 6 with the treatment-effect columns)
    t_log = time.perf_counter()
    model = PPO.load(str(model_path), device="cpu")
    head = xme.build_head(model.policy, args.target)
    f = xme.head_fn(head)
    reference = states_pool.mean(axis=0)

    # the paper's wait-state pool for the wait margin; the whole pool otherwise
    if args.target == "wait_margin":
        eval_states = states_pool[f(states_pool) >= 0]
    elif args.target == "margin":
        eval_states = states_pool[f(states_pool) > 0]
    else:
        eval_states = states_pool
    if args.quick and len(eval_states) > args.quick:
        eval_states = eval_states[np.random.default_rng(pools.POOL_SEED).choice(len(eval_states), args.quick, replace=False)]
    n = len(eval_states)
    print(f"\n=== {name}: target={args.target} pool={len(states_pool)} explained={n}")
    out: dict = {
        "log": name, "variant": args.variant, "model": str(model_path), "target": args.target, "feature_names": FEATS, "families": xm.FAMILIES,
        "n_pool_states": int(len(states_pool)), "n_explained_states": int(n), "reference": reference.tolist(),
        "target_mean": float(f(eval_states).mean()), "target_std": float(f(eval_states).std()),
    }
    if n < 5:
        out["skipped"] = "fewer than 5 states to explain"
        return out
    figdir = paths.XAI_FIGURES / name
    figdir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    # --- local attributions (phi per state) --------------------------------
    atts: dict[str, xme.Attribution] = {}
    atts["integrated_gradients"] = xme.ig_attribution(head, eval_states, reference)
    atts["kernel_shap"] = xme.kernel_shap(head, eval_states, states_pool, seed=args.seed)
    for label, fn in (("deep_shap", xme.deep_shap), ("gradient_shap", xme.gradient_shap)):
        try:
            atts[label] = fn(head, eval_states, states_pool, seed=args.seed)
        except Exception as e:  # architecture-sensitive; record rather than abort
            out.setdefault("errors", {})[label] = f"{type(e).__name__}: {e}"
    n_lime = min(n, args.n_lime)
    lime_states = eval_states[:n_lime]
    atts["lime"] = xme.lime_attribution(head, lime_states, states_pool, FEATS, num_samples=args.lime_samples, discretize=args.lime_discretize, seed=args.seed)
    for label, a in atts.items():
        print(f"  {label:22s} {a.seconds:7.1f}s  global |phi| share = "
              + ", ".join(f"{fn}={s:.1%}" for fn, s in zip(FEATS, a.global_importance / a.global_importance.sum())))

    # --- global, model-side quantities ------------------------------------
    perm = xme.permutation_importance(head, eval_states, n_repeats=10, seed=args.seed)
    vimp = xme.var_importance(head, eval_states, seed=args.seed)
    model_effects = np.asarray(vimp["effects"])
    fc = xm.functional_complexity(f, eval_states, FEATS, seed=args.seed)
    ale = xme.ale_all(head, eval_states, FEATS)
    crit = xm.criteria_scores(f, eval_states, FEATS, model_effects, seed=args.seed)
    rc = xm.reducts_and_core(crit["criteria"], FEATS, seed=args.seed)
    print(f"  permutation effects: {dict(zip(FEATS, np.round(perm['importances_mean'], 3)))}")
    print(f"  var_importance      : {dict(zip(FEATS, np.round(model_effects, 3)))}")
    print(f"  functional complexity per feature (flip / L1): "
          + ", ".join(f"{k}={v['flip']:.3f}/{v['l1']:.1f}" for k, v in fc["per_feature"].items()))
    print(f"  ConsisXAI core={rc['core']} reduct={rc['selected_reduct']} (threshold {rc['threshold']:.2f}, {rc['attempts']} redraws)")
    out.update({
        "permutation_importance": perm, "var_importance": vimp, "functional_complexity": fc, "ale": ale,
        "consisxai_criteria": crit, "consisxai_reducts_core": rc,
    })

    # --- per-method explanation metrics -----------------------------------
    per_method = {}
    importances = {}
    for label, a in atts.items():
        g = a.global_importance
        importances[label] = g
        st = eval_states if label != "lime" else lime_states
        dl = deletion_test(head, st, a.phi, reference, k=1, n_random=20, seed=args.seed, track_sign_flips=True)
        per_method[label] = {
            "seconds": a.seconds,
            "expected_value": a.expected_value,
            "global_importance": g.tolist(),
            "global_share": (g / g.sum()).tolist() if g.sum() > 0 else g.tolist(),
            "mean_signed": a.phi.mean(axis=0).tolist(),
            "top1_feature_counts": {fn: int((np.argmax(np.abs(a.phi), axis=1) == i).sum()) for i, fn in enumerate(FEATS)},
            "parsimony": xm.parsimony(g, FEATS, share_threshold=args.share_threshold),
            "monotonicity": xm.monotonicity(g, model_effects),
            "monotonicity_vs_permutation": xm.monotonicity(g, np.asarray(perm["importances_mean"])),
            "lod": xm.lod(g, model_effects, FEATS, k=2),
            "consisxai": xm.consisxai(g, FEATS, rc, n=n),
            "deletion_test_k1": {**dl.as_dict(), "z": float(dl.gap / dl.gap_se) if dl.gap_se > 0 else None},
            **a.extras,
        }
        m = per_method[label]
        print(f"  {label:22s} parsimony={m['parsimony']['above_share']} rho={m['monotonicity']['spearman']:+.2f} "
              f"tau={m['monotonicity']['kendall']:+.2f} LOD={m['lod']['lod']:.2f} "
              f"core-ratio={m['consisxai']['core']['experimental']:.2f} reduct-ratio={m['consisxai']['reduct']['experimental']:.2f} "
              f"deletion z={m['deletion_test_k1']['z']:.1f}")
    out["methods"] = per_method
    out["agreement"] = xm.agreement(importances, FEATS, k=2)

    # --- local instances ---------------------------------------------------
    loc = local_instances(eval_states, seed=args.seed)
    picks = dict(loc["picks"])
    # the paper's cards (figures/make_figures.py): median act / median wait decision of the pool
    m_all = xme.head_fn(xme.build_head(model.policy, "margin"))(eval_states)
    picks.update({f"paper_card_{k}": v for k, v in pools.paper_card_indices(eval_states, m_all).items()})
    local = {}
    lime_ex = xme.lime_explainer(states_pool, FEATS, discretize=args.lime_discretize, seed=args.seed)
    for tag, i in picks.items():
        s = eval_states[i]
        entry = {"index": int(i), "state": dict(zip(FEATS, s.astype(float).tolist())), "target": float(f(s[None])[0])}
        for label, a in atts.items():
            if i < len(a.phi):
                entry[label] = dict(zip(FEATS, a.phi[i].tolist()))
        lexp = lime_ex.explain_instance(s.astype(np.float64), f, num_features=len(FEATS), num_samples=args.lime_samples)
        entry["lime_as_list"] = lexp.as_list()
        entry["lime_local_r2"] = float(lexp.score)
        entry["lime_stability"] = xme.lime_stability(head, s, states_pool, FEATS, n_calls=args.stability_calls, num_samples=args.lime_samples, discretize=args.lime_discretize, seed=args.seed)
        local[tag] = entry
        if not args.skip_plots:
            files += xp.lime_local_plot(lexp, i, figdir, f"{name}_{tag}")
            for label in ("kernel_shap", "integrated_gradients"):
                files += xp.shap_local_plots(xme.shap_explanation(atts[label], eval_states, FEATS), i, figdir, f"{name}_{tag}_{label}")
    out["local_instances"] = {"selection": loc, "instances": local}
    print("  local instances:", {t: f"csi={e['lime_stability']['csi']:.0f} vsi={e['lime_stability']['vsi']:.0f}" for t, e in local.items()})

    # --- figures -----------------------------------------------------------
    if not args.skip_plots:
        for label in ("kernel_shap", "deep_shap", "integrated_gradients"):
            if label in atts:
                files += xp.shap_global_plots(xme.shap_explanation(atts[label], eval_states, FEATS), figdir, f"{name}_{label}")
        files += xp.permutation_plots(perm, FEATS, figdir, name)
        files += xp.ale_plots(ale, FEATS, figdir, name)
        files += xp.correlation_heatmap(eval_states, FEATS, figdir, name)
        files += xp.mutual_info_plot(np.asarray(crit["criteria"]["mutual_information"]), FEATS, figdir, name)
        files += xp.importance_comparison(importances, FEATS, figdir, name)
        files += xp.agreement_heatmap(out["agreement"], figdir, name, key="spearman")
        files += xp.agreement_heatmap(out["agreement"], figdir, name, key="jaccard_top2")
    out["figures"] = files
    out["seconds_total"] = time.perf_counter() - t_log
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", default=LOGS, choices=LOGS)
    ap.add_argument("--variant", default=pools.DEFAULT_VARIANT, choices=list(pools.VARIANTS),
                    help="policy variant (all three logs); default = the paper's policy")
    ap.add_argument("--target", default="wait_margin", choices=xme.TARGETS)
    ap.add_argument("--seed", type=int, default=pools.POOL_SEED)
    ap.add_argument("--n-lime", type=int, default=500, help="states explained with LIME (5000 samples each)")
    ap.add_argument("--lime-samples", type=int, default=5000)
    ap.add_argument("--lime-discretize", action="store_true", help="LIME's discretised surrogate (the reference default; poorer local fit here)")
    ap.add_argument("--stability-calls", type=int, default=10, help="LIME repetitions for CSI/VSI")
    ap.add_argument("--share-threshold", type=float, default=0.05, help="parsimony: minimum share of |importance| to count a feature")
    ap.add_argument("--quick", type=int, default=0, help="explain at most this many states (smoke run)")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--out", default=None, help="JSON path (default paths.XAI_JSON)")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    results = []
    for name in args.logs:
        states, _rows, feats = pools.evaluation_pool(name, args.variant)
        results.append(run_log(name, paths.variant_model(name, args.variant), states, args, feats))
    out_path = Path(args.out) if args.out else (paths.XAI_JSON if args.variant == pools.DEFAULT_VARIANT
                                                 else paths.REPO / f"xai_suite_results_{args.variant}.json")
    out_path.write_text(json.dumps(_jsonable({"args": vars(args), "results": results}), indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
