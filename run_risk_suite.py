"""The risk explanation -- *why is this case predicted to end badly?* -- on
the same decision points the timing justification is read on, and its
evaluation with the metrics of the outcome-prediction explainability
literature (Stevens & De Smedt, Stevens et al., Elkhawaga et al.) plus the
paper's own deletion test.

Risk score: r(x_t) = P(deviant | prefix x_t) from the retrained CatBoost
outcome predictor (risk_model.py), the model whose output the policy's
``deviation``/``reliability`` features are derived from. Attributions are
SHAP TreeExplainer values of the log-odds of r (the literature's setting for
tree models), on the prefixes of the Section 6/7 evaluation pool matched by
case id and prefix number, so each pool row has both a risk card and a
timing card.

Per log this writes to paths.RISK_JSON / paths.RISK_FIGURES/<log>/:
  * global SHAP bar + beeswarm over the pool's prefixes, per-family shares;
  * parsimony, functional complexity (per family, L1 and flip), monotonicity
    (Spearman/Kendall vs. observed-value permutation effects), LOD (k=10,
    real event / case / control-flow families), all as in Stevens & De Smedt;
  * the paper's deletion test on the risk log-odds (guided / random / anti,
    k = 1, 3), masking to the pool reference (numeric mean, categorical mode);
  * two-level cards for chosen decision points: risk top features (SHAP)
    next to the timing justification (IG on the wait margin);
  * the link between the two levels on the pool: r vs. the state's
    ``deviation``/``reliability``, and the retrained r vs. the shipped one.

Usage: python run_risk_suite.py [--logs BPIC2017] [--quick 60] [--skip-plots]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

import paths  # noqa: E402
import pools  # noqa: E402
import risk_model as rm  # noqa: E402
import xai_metrics as xm  # noqa: E402
from dual_level import MarginHead, WaitMarginHead, integrated_gradients  # noqa: E402
from run_xai_suite import _jsonable  # noqa: E402

LOGS = list(pools.LOGS)


# ---------------------------------------------------------------------------
# Black box over a CatBoost model + DataFrame prefixes
# ---------------------------------------------------------------------------


class RiskBox:
    """f(X) -> log-odds of P(deviant); X a DataFrame in the model's columns."""

    def __init__(self, clf, feats: dict):
        self.clf = clf
        self.columns = feats["feature_names"]
        self.cat_idx = feats["cat_feature_indices"]
        self.cat_cols = [self.columns[i] for i in self.cat_idx]

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X[self.columns])[:, 1]

    def logodds(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.proba(X), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def reference(self, X: pd.DataFrame) -> pd.Series:
        ref = {}
        for c in self.columns:
            ref[c] = X[c].mode().iloc[0] if c in self.cat_cols else float(X[c].astype(float).mean())
        return pd.Series(ref)

    def mask(self, X: pd.DataFrame, idx: np.ndarray, ref: pd.Series) -> pd.DataFrame:
        """Row i gets columns idx[i] set to the reference value."""
        out = X.copy()
        for j, c in enumerate(self.columns):
            rows = np.where((idx == j).any(axis=1))[0]
            if len(rows):
                out.iloc[rows, out.columns.get_loc(c)] = ref[c]
        return out


def deletion_test_df(box: RiskBox, X: pd.DataFrame, phi: np.ndarray, ref: pd.Series, k: int, n_random: int = 20, seed: int = 123) -> dict:
    """dual_level.deletion_test for a DataFrame black box (mixed types)."""
    rng = np.random.default_rng(seed)
    f0 = box.logodds(X)
    order = np.argsort(-np.abs(phi), axis=1)
    guided, anti = order[:, :k], order[:, -k:]
    fg = box.logodds(box.mask(X, guided, ref))
    fa = box.logodds(box.mask(X, anti, ref))
    D = X.shape[1]
    rand = np.zeros((n_random, len(X)))
    for r in range(n_random):
        ridx = np.stack([rng.choice(D, size=k, replace=False) for _ in range(len(X))])
        rand[r] = np.abs(box.logodds(box.mask(X, ridx, ref)) - f0)
    dg, dr, da = np.abs(fg - f0), rand.mean(axis=0), np.abs(fa - f0)
    paired = dg - dr
    se = float(paired.std(ddof=1) / np.sqrt(len(paired)))
    return {"k": k, "abs_guided": float(dg.mean()), "abs_random": float(dr.mean()), "abs_anti": float(da.mean()),
            "gap": float(paired.mean()), "gap_se": se, "z": float(paired.mean() / se) if se > 0 else None,
            "flip_guided": float((np.sign(fg) != np.sign(f0)).mean())}


def replace_other_observed_df(X: pd.DataFrame, cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    out = X.copy()
    for c in cols:
        vals = X[c].to_numpy()
        uniq = pd.unique(vals)
        if len(uniq) < 2:
            continue
        new = np.empty_like(vals)
        for i, v in enumerate(vals):
            others = uniq[uniq != v]
            new[i] = rng.choice(others)
        out[c] = new
    return out


def var_importance_df(box: RiskBox, X: pd.DataFrame, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f0 = box.logodds(X)
    eff = np.zeros(len(box.columns))
    for j, c in enumerate(box.columns):
        eff[j] = float(np.sqrt(np.mean((box.logodds(replace_other_observed_df(X, [c], rng)) - f0) ** 2)))
    return eff


def functional_complexity_df(box: RiskBox, X: pd.DataFrame, families: dict, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    p0 = box.proba(X)
    out = {}
    for fam, cols in families.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            out[fam] = {"l1": 0.0, "flip": 0.0, "n_columns": 0}
            continue
        p1 = box.proba(replace_other_observed_df(X, cols, rng))
        out[fam] = {"l1": float(100 * np.mean(np.abs(p1 - p0))), "flip": float(np.mean((p1 > 0.5) != (p0 > 0.5))), "n_columns": len(cols)}
    return out


def lod_families(g: np.ndarray, effects: np.ndarray, names: list[str], families: dict, k: int = 10) -> dict:
    fam_of = {c: f for f, cols in families.items() for c in cols}
    order = list(families)

    def counts(v):
        top = [names[i] for i in np.argsort(-np.abs(v), kind="stable")[:k]]
        return [sum(1 for t in top if fam_of.get(t) == f) for f in order]

    ca, cb = counts(g), counts(effects)
    return {"k": k, "families": order, "explanation_counts": ca, "model_counts": cb, "lod": float(np.linalg.norm(np.array(ca) - np.array(cb)))}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save(fig, out: Path, stem: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    ps = []
    for ext in ("pdf", "png"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        ps.append(str(p))
    plt.close("all")
    return ps


def two_level_card(risk_phi: np.ndarray, risk_names: list[str], risk_values: dict, r: float,
                   timing_phi: np.ndarray, timing_names: list[str], dq: float, acts: bool, out: Path, stem: str, top: int = 6) -> list[str]:
    """Left: why the case is at risk (top SHAP features of the risk log-odds);
    right: why the policy acts now (IG on Delta Q) or waits (IG on Delta Q_wait)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    o = np.argsort(-np.abs(risk_phi))[:top]
    labels = [f"{risk_names[i]} = {risk_values[risk_names[i]]}" if not isinstance(risk_values[risk_names[i]], float)
              else f"{risk_names[i]} = {risk_values[risk_names[i]]:.3g}" for i in o]
    ax1.barh(range(len(o)), risk_phi[o], color=["#c44e52" if v > 0 else "#4c72b0" for v in risk_phi[o]])
    ax1.set_yticks(range(len(o)), labels, fontsize=7.5)
    ax1.invert_yaxis()
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_xlabel(r"$\phi^{r}$ (pushes toward a bad outcome $\rightarrow$)", fontsize=8)
    ax1.set_title(f"Why at risk?  $r$ = P(deviant) = {r:.2f}", fontsize=9)
    o2 = np.argsort(-np.abs(timing_phi))
    ax2.barh(range(len(o2)), timing_phi[o2], color=["#c44e52" if v > 0 else "#4c72b0" for v in timing_phi[o2]])
    ax2.set_yticks(range(len(o2)), [timing_names[i] for i in o2], fontsize=8)
    ax2.invert_yaxis()
    ax2.axvline(0, color="black", lw=0.8)
    if acts:
        ax2.set_xlabel(r"$\phi^{\Delta Q}$ (pushes toward acting now $\rightarrow$)", fontsize=8)
        ax2.set_title(f"Why act now?  $\\Delta Q$ = {dq:.2f}", fontsize=9)
    else:
        ax2.set_xlabel(r"$\phi^{\Delta Q_{\mathrm{wait}}}$ (pulls toward waiting $\rightarrow$)", fontsize=8)
        ax2.set_title(f"Why wait now?  $\\Delta Q_{{wait}}$ = {-dq:.2f}", fontsize=9)
    fig.tight_layout()
    return _save(fig, out, stem)


# ---------------------------------------------------------------------------
# Main per-log routine
# ---------------------------------------------------------------------------


def run_log(name: str, args) -> dict:
    import shap

    t0 = time.perf_counter()
    model_path = paths.variant_model(name, args.variant)
    states, rows, sfeats = pools.evaluation_pool(name, args.variant)  # sfeats: the policy's state features
    clf, feats = rm.load_model(name)
    box = RiskBox(clf, feats)
    families = feats["families"]
    X, meta = rm.prefixes_for_pool(name, rows)
    keep = meta["pool_row"].to_numpy()
    states_m, rows_m = states[keep], rows.iloc[keep].reset_index(drop=True)
    if args.quick and len(X) > args.quick:
        sel = np.random.default_rng(pools.POOL_SEED).choice(len(X), args.quick, replace=False)
        X, meta, states_m, rows_m = X.iloc[sel].reset_index(drop=True), meta.iloc[sel].reset_index(drop=True), states_m[sel], rows_m.iloc[sel].reset_index(drop=True)
    n = len(X)
    print(f"\n=== {name}: pool rows matched {n}/{len(rows)}; {X.shape[1]} risk features")
    out: dict = {"log": name, "n_pool_rows": int(len(rows)), "n_matched": int(n), "n_risk_features": int(X.shape[1]),
                 "families_sizes": {f: len(c) for f, c in families.items()}, "risk_manifest": json.loads(rm.model_paths(name)["manifest"].read_text())}
    figdir = paths.RISK_FIGURES / name
    files: list[str] = []

    # --- risk score and its SHAP attribution ------------------------------
    r = box.proba(X)
    explainer = shap.TreeExplainer(clf)
    phi = np.asarray(explainer.shap_values(X[box.columns]), dtype=np.float64)
    if phi.ndim == 3:
        phi = phi[..., 1] if phi.shape[-1] == 2 else phi[1]
    ev = float(np.ravel(explainer.expected_value)[-1])
    g = np.abs(phi).mean(axis=0)
    share = g / g.sum()
    fam_share = {f: float(sum(share[box.columns.index(c)] for c in cols if c in box.columns)) for f, cols in families.items()}
    top10 = [(box.columns[i], float(share[i])) for i in np.argsort(-g)[:10]]
    out["risk"] = {"mean": float(r.mean()), "std": float(r.std()), "share_above_0_5": float((r > 0.5).mean()), "expected_value_logodds": ev,
                   "global_share_top10": top10, "family_share": fam_share, "deviant_share_in_pool": float(meta.y.mean())}
    print(f"  r: mean={r.mean():.3f}, P(r>0.5)={(r > 0.5).mean():.3f}; top features: " + ", ".join(f"{k}={v:.1%}" for k, v in top10[:5]))
    print(f"  family share: {fam_share}")

    # --- link to the policy's state --------------------------------------
    st = pd.DataFrame(states_m, columns=sfeats)
    link = {"corr_r_deviation": float(np.corrcoef(r, st.deviation)[0, 1]) if st.deviation.nunique() > 1 else None,
            "corr_r_reliability": float(np.corrcoef(r, st.reliability)[0, 1]) if st.reliability.nunique() > 1 else None,
            "deviation_equals_1_minus_predicted": float(((r < 0.5).astype(float) == st.deviation).mean()) if name != "SimBank" else None,
            "reliability_equals_deviation": bool((st.reliability == st.deviation).all())}
    if "predicted_proba_1" in rows.columns:
        shipped = rows["predicted_proba_1"].to_numpy()[keep]
        if args.quick and len(shipped) > args.quick:
            shipped = shipped[sel]
        link["corr_retrained_vs_shipped_r"] = float(np.corrcoef(r, shipped)[0, 1])
        link["mean_shipped_r"] = float(shipped.mean())
    out["link_to_state"] = link
    print(f"  link: {link}")

    # --- literature metrics ------------------------------------------------
    ref = box.reference(X)
    effects = var_importance_df(box, X, seed=args.seed)
    fc = functional_complexity_df(box, X, families, seed=args.seed)
    rho, p_rho = spearmanr(g, effects)
    tau, p_tau = kendalltau(g, effects)
    nonzero = int((g > 1e-9).sum())
    pars_fam = {f: int(sum(1 for c in cols if c in box.columns and g[box.columns.index(c)] > 1e-9)) for f, cols in families.items()}
    metrics = {
        "parsimony": {"nonzero": nonzero, "of": int(len(g)), "by_family": pars_fam, "above_1pct_share": int((share >= 0.01).sum()),
                      "effective": float(np.exp(-(share[share > 0] * np.log(share[share > 0])).sum()))},
        "functional_complexity": fc,
        "monotonicity": {"spearman": float(rho), "spearman_p": float(p_rho), "kendall": float(tau), "kendall_p": float(p_tau)},
        "lod": lod_families(g, effects, box.columns, families, k=10),
        "deletion_test": {str(k): deletion_test_df(box, X, phi, ref, k, seed=args.seed) for k in (1, 3)},
    }
    out["metrics"] = metrics
    print(f"  parsimony={nonzero}/{len(g)} rho={rho:+.2f} tau={tau:+.2f} LOD={metrics['lod']['lod']:.2f} "
          f"FC={ {f: round(v['flip'], 3) for f, v in fc.items()} } deletion z(k=1)={metrics['deletion_test']['1']['z']:.1f} z(k=3)={metrics['deletion_test']['3']['z']:.1f}")

    # --- two-level cards ---------------------------------------------------
    ppo = PPO.load(str(model_path), device="cpu")
    margin = MarginHead(ppo.policy, intervene_action=1)
    wait_head = WaitMarginHead(margin)
    reference_state = states.mean(axis=0)
    picks = {"highest_risk": int(np.argmax(r)), "lowest_risk": int(np.argmin(r))}
    with torch.no_grad():
        m_all = margin(torch.from_numpy(states_m)).numpy()
    picks.update({f"paper_card_{k}": v for k, v in pools.paper_card_indices(states_m, m_all, rows_m).items()})
    oracle = pools.oracle_acts(rows_m)
    out["policy_on_pool"] = {"intervene_rate": float((m_all > 0).mean()), "oracle_positive_rate": float(oracle.mean()),
                             "agreement_with_oracle": float(((m_all > 0) == oracle).mean())}
    cards = {}
    for tag, i in picks.items():
        s = states_m[i:i + 1]
        with torch.no_grad():
            dq = float(margin(torch.from_numpy(s)).numpy()[0])
        acts = dq > 0  # attribute the side the policy chose: Delta Q when it acts, Delta Q_wait when it waits
        phi_t = integrated_gradients(margin if acts else wait_head, s, reference_state, n_steps=128)[0]
        vals = {c: (X.iloc[i][c] if c in box.cat_cols else float(X.iloc[i][c])) for c in box.columns}
        o = np.argsort(-np.abs(phi[i]))[:6]
        cards[tag] = {"pool_row": int(meta.pool_row.iloc[i]), "case_id": str(meta.case_id.iloc[i]), "prefix_nr": int(meta.prefix_nr.iloc[i]),
                      "actual_deviant": int(meta.y.iloc[i]), "r": float(r[i]), "state": dict(zip(sfeats, s[0].astype(float).tolist())),
                      "action": "intervene" if acts else "wait", "dq": dq, "dq_wait": -dq,
                      "risk_top": [(box.columns[j], vals[box.columns[j]], float(phi[i, j])) for j in o],
                      "timing_phi": dict(zip(sfeats, phi_t.astype(float).tolist()))}
        if not args.skip_plots:
            files += two_level_card(phi[i], box.columns, vals, float(r[i]), phi_t, sfeats, dq, acts, figdir, f"{name}_{tag}_two_level_card")
    out["cards"] = cards

    # --- figures -----------------------------------------------------------
    if not args.skip_plots:
        Xnum = X.copy()
        for c in box.cat_cols:  # beeswarm needs numeric colouring; categorical codes
            Xnum[c] = pd.Categorical(Xnum[c]).codes
        expl = shap.Explanation(values=phi, base_values=np.full(n, ev), data=Xnum[box.columns].to_numpy(dtype=float), feature_names=box.columns)
        plt.figure(figsize=(6.5, 4.5)); shap.plots.beeswarm(expl, max_display=12, show=False); files += _save(plt.gcf(), figdir, f"{name}_risk_beeswarm")
        plt.figure(figsize=(6.5, 4.0)); shap.plots.bar(expl, max_display=12, show=False); files += _save(plt.gcf(), figdir, f"{name}_risk_bar")
        fig, ax = plt.subplots(figsize=(4.2, 2.6))
        ax.bar(list(fam_share), list(fam_share.values()), color=["#4c72b0", "#c44e52", "#2e8b57"])
        ax.set_ylabel("share of mean |SHAP|"); ax.set_title(f"{name}: risk attribution by attribute family", fontsize=9)
        files += _save(fig, figdir, f"{name}_risk_family_share")
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        ax.scatter(r, st.deviation + np.random.default_rng(0).normal(0, 0.02, n), s=6, alpha=0.4)
        ax.set_xlabel("retrained r = P(deviant)"); ax.set_ylabel("state feature deviation (jittered)")
        files += _save(fig, figdir, f"{name}_r_vs_deviation")
    out["figures"] = files
    out["seconds"] = time.perf_counter() - t0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", default=LOGS, choices=LOGS)
    ap.add_argument("--variant", default=pools.DEFAULT_VARIANT, choices=list(pools.VARIANTS),
                    help="policy variant (all three logs); default = the paper's policy")
    ap.add_argument("--seed", type=int, default=pools.POOL_SEED)
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    results = [run_log(name, args) for name in args.logs]
    out_path = Path(args.out) if args.out else (paths.RISK_JSON if args.variant == pools.DEFAULT_VARIANT
                                                 else paths.REPO / f"risk_results_{args.variant}.json")
    out_path.write_text(json.dumps(_jsonable({"args": vars(args), "results": results}), indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
