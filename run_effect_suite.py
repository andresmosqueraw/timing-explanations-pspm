"""The effect explanation -- *why does the estimator expect the intervention
to change this case's outcome?* -- on the same decision points as the risk
and timing explanations, plus the three-level card that puts them together.

Effect score: CATE(x_t) = p_U(x_t) - p_T(x_t) (the drop in P(undesired) the
treatment buys) from the retrained two-model
estimator (effect_model.py), whose two probabilities the policy reads as
its state features Proba_if_Treated / Proba_if_Untreated. Attribution:
TreeSHAP on each arm's log-odds, summed back from the one-hot columns to the
prefix attribute, and the effect attribution as the *difference* of the two
arms' attributions (attribute-then-subtract, as Padella et al. do for a
recommendation's before/after KPI; here the two arms are the two
counterfactual outcomes of one prefix):

    phi^{p_T}, phi^{p_U} : TreeSHAP of logit p_T, logit p_U
    phi^{CATE}           : phi^{p_U} - phi^{p_T}   (sums to the log-odds-ratio
                            difference between the two arms, minus its pool mean)

Fidelity: the paper's deletion test on the *effect* -- mask the k prefix
attributes phi^{CATE} ranks highest to the pool reference and measure how
far logit p_U - logit p_T moves, against random and anti-guided masking.

Per BPIC log this writes to paths.EFFECT_JSON / paths.EFFECT_FIGURES/<log>/:
global shares per attribute and family, the deletion test (k = 1, 3), the
link between the retrained and the shipped probabilities on the pool, and,
for the paper's decision points, the three-level card (risk | effect |
timing). SimBank has no effect model (its effect feature is the
simulator's own uncertainty) and is skipped.

Usage: python run_effect_suite.py [--logs BPIC2017] [--quick 60] [--skip-plots]
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
from stable_baselines3 import PPO  # noqa: E402

import compose as cp  # noqa: E402
import effect_model as em  # noqa: E402
import paths  # noqa: E402
import pools  # noqa: E402
import risk_model as rm  # noqa: E402
from dual_level import MarginHead, WaitMarginHead, integrated_gradients  # noqa: E402
from run_risk_suite import RiskBox, _save, replace_other_observed_df  # noqa: E402
from run_xai_suite import _jsonable  # noqa: E402

LOGS = list(em.EFFECT_LOGS)


class EffectBox:
    """Black box over the two arms: p_T, p_U and the effect on raw prefixes."""

    def __init__(self, arms, feats):
        self.arms = arms
        self.columns = feats["columns"]          # one-hot columns
        self.raw_columns = feats["raw_columns"]  # prefix attributes
        self.cat_cols = feats["cat_cols"]
        self.families = feats["families"]

    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        return em.one_hot(X[self.raw_columns], self.cat_cols, self.columns)

    def probs(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xd = self.encode(X)
        return self.arms["treated"].predict_proba(Xd)[:, 1], self.arms["untreated"].predict_proba(Xd)[:, 1]

    def logodds_effect(self, X: pd.DataFrame) -> np.ndarray:
        pT, pU = self.probs(X)
        pT, pU = np.clip(pT, 1e-6, 1 - 1e-6), np.clip(pU, 1e-6, 1 - 1e-6)
        return np.log(pU / (1 - pU)) - np.log(pT / (1 - pT))

    def reference(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series({c: (X[c].mode().iloc[0] if c in self.cat_cols else float(X[c].astype(float).mean())) for c in self.raw_columns})

    def mask(self, X: pd.DataFrame, idx: np.ndarray, ref: pd.Series) -> pd.DataFrame:
        out = X[self.raw_columns].copy()
        for j, c in enumerate(self.raw_columns):
            rows = np.where((idx == j).any(axis=1))[0]
            if len(rows):
                out.iloc[rows, out.columns.get_loc(c)] = ref[c]
        return out

    def shap_raw(self, X: pd.DataFrame, background: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """TreeSHAP per arm on the one-hot matrix, summed back to raw attributes.
        With ``background`` (raw prefixes) the explainer is interventional
        against that sample, so each arm's attribution sums to its log-odds
        minus the background's expectation (compose.py's reference alignment);
        without it, the tree-path-dependent default."""
        import shap

        Xd = self.encode(X)
        bg = None if background is None else self.encode(background)
        out, ev = [], {}
        # map one-hot column -> raw attribute
        owner = {}
        for col in self.columns:
            owner[col] = next((c for c in self.cat_cols if col.startswith(c + "_")), col)
        raw_idx = {c: i for i, c in enumerate(self.raw_columns)}
        for arm in ("treated", "untreated"):
            if isinstance(self.arms[arm], em.ConstantArm):  # constant arm: zero attribution
                p = float(np.clip(self.arms[arm].p, 1e-6, 1 - 1e-6))
                out.append(np.zeros((len(X), len(self.raw_columns)))); ev[arm] = float(np.log(p / (1 - p)))
                continue
            ex = shap.TreeExplainer(self.arms[arm]) if bg is None else shap.TreeExplainer(self.arms[arm], data=bg, feature_perturbation="interventional")
            sv = np.asarray(ex.shap_values(Xd), dtype=np.float64)
            if sv.ndim == 3:
                sv = sv[..., -1]
            agg = np.zeros((len(X), len(self.raw_columns)))
            for k, col in enumerate(self.columns):
                agg[:, raw_idx[owner[col]]] += sv[:, k]
            out.append(agg)
            ev[arm] = float(np.ravel(ex.expected_value)[-1])
        return out[0], out[1], ev


def deletion_test_effect(box: EffectBox, X: pd.DataFrame, phi: np.ndarray, ref: pd.Series, k: int, n_random: int = 20, seed: int = 123) -> dict:
    rng = np.random.default_rng(seed)
    f0 = box.logodds_effect(X)
    order = np.argsort(-np.abs(phi), axis=1)
    fg = box.logodds_effect(box.mask(X, order[:, :k], ref))
    fa = box.logodds_effect(box.mask(X, order[:, -k:], ref))
    D = len(box.raw_columns)
    rand = np.zeros((n_random, len(X)))
    for r in range(n_random):
        ridx = np.stack([rng.choice(D, size=k, replace=False) for _ in range(len(X))])
        rand[r] = np.abs(box.logodds_effect(box.mask(X, ridx, ref)) - f0)
    dg, dr, da = np.abs(fg - f0), rand.mean(axis=0), np.abs(fa - f0)
    paired = dg - dr
    se = float(paired.std(ddof=1) / np.sqrt(len(paired)))
    # does masking flip the reward's positive-effect rule y1 - y0 > 0 ?
    pT0, pU0 = box.probs(X)
    pTg, pUg = box.probs(box.mask(X, order[:, :k], ref))
    rule0, ruleg = cp.positive_effect_rule(pT0, pU0), cp.positive_effect_rule(pTg, pUg)
    return {"k": k, "abs_guided": float(dg.mean()), "abs_random": float(dr.mean()), "abs_anti": float(da.mean()),
            "gap": float(paired.mean()), "gap_se": se, "z": float(paired.mean() / se) if se > 0 else None,
            "flip_positive_effect_rule": float((rule0 != ruleg).mean())}


def three_level_card(risk_phi, risk_names, risk_vals, r, eff_phi, eff_names, eff_vals, pT, pU,
                     timing_phi, timing_names, dq, acts, out: Path, stem: str, top: int = 6) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.0), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    panels = [
        (axes[0], risk_phi, risk_names, risk_vals, f"Why at risk?  $r$ = P(deviant) = {r:.2f}", r"$\phi^{r}$ (toward a bad outcome $\rightarrow$)"),
        (axes[1], eff_phi, eff_names, eff_vals, f"Why would acting help?  $\\hat p_T$ = {pT:.2f}, $\\hat p_U$ = {pU:.2f}", r"$\phi^{\mathrm{CATE}}$ (toward a larger effect $\rightarrow$)"),
    ]
    for ax, phi, names, vals, title, xlabel in panels:
        o = np.argsort(-np.abs(phi))[:top]
        labels = [f"{names[i]} = {vals[names[i]]}" if not isinstance(vals[names[i]], float) else f"{names[i]} = {vals[names[i]]:.3g}" for i in o]
        ax.barh(range(len(o)), phi[o], color=["#c44e52" if v > 0 else "#4c72b0" for v in phi[o]])
        ax.set_yticks(range(len(o)), labels, fontsize=7.5)
        ax.invert_yaxis(); ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel(xlabel, fontsize=8); ax.set_title(title, fontsize=9)
    ax = axes[2]
    o = np.argsort(-np.abs(timing_phi))
    ax.barh(range(len(o)), timing_phi[o], color=["#c44e52" if v > 0 else "#4c72b0" for v in timing_phi[o]])
    ax.set_yticks(range(len(o)), [timing_names[i] for i in o], fontsize=8)
    ax.invert_yaxis(); ax.axvline(0, color="black", lw=0.8)
    if acts:
        ax.set_xlabel(r"$\phi^{\Delta Q}$ (toward acting now $\rightarrow$)", fontsize=8); ax.set_title(f"Why act now?  $\\Delta Q$ = {dq:.2f}", fontsize=9)
    else:
        ax.set_xlabel(r"$\phi^{\Delta Q_{\mathrm{wait}}}$ (toward waiting $\rightarrow$)", fontsize=8); ax.set_title(f"Why wait now?  $\\Delta Q_{{wait}}$ = {-dq:.2f}", fontsize=9)
    fig.tight_layout()
    return _save(fig, out, stem)


def run_log(name: str, args) -> dict:
    t0 = time.perf_counter()
    states, rows, sfeats = pools.evaluation_pool(name, args.variant)
    arms, feats = em.load_model(name)
    box = EffectBox(arms, feats)
    X, meta = rm.prefixes_for_pool(name, rows)
    keep = meta["pool_row"].to_numpy()
    states_m, rows_m = states[keep], rows.iloc[keep].reset_index(drop=True)
    if args.quick and len(X) > args.quick:
        sel = np.random.default_rng(pools.POOL_SEED).choice(len(X), args.quick, replace=False)
        X, meta, states_m, rows_m = X.iloc[sel].reset_index(drop=True), meta.iloc[sel].reset_index(drop=True), states_m[sel], rows_m.iloc[sel].reset_index(drop=True)
    n = len(X)
    print(f"\n=== {name}: pool rows matched {n}/{len(rows)}; {len(box.raw_columns)} prefix attributes")
    out: dict = {"log": name, "n_matched": int(n), "manifest": json.loads(em.model_paths(name)["manifest"].read_text())}
    figdir = paths.EFFECT_FIGURES / name
    files: list[str] = []

    pT, pU = box.probs(X)
    phiT, phiU, ev = box.shap_raw(X)
    phi = phiU - phiT
    g = np.abs(phi).mean(axis=0)
    share = g / g.sum()
    fam_share = {f: float(sum(share[box.raw_columns.index(c)] for c in cols if c in box.raw_columns)) for f, cols in box.families.items()}
    top10 = [(box.raw_columns[i], float(share[i])) for i in np.argsort(-g)[:10]]
    out["effect"] = {"mean_pT": float(pT.mean()), "mean_pU": float(pU.mean()), "mean_cate": float((pU - pT).mean()),
                     "share_positive_rule": float(cp.positive_effect_rule(pT, pU).mean()), "expected_value_logodds": ev,
                     "global_share_top10": top10, "family_share": fam_share}
    print(f"  pT={pT.mean():.3f} pU={pU.mean():.3f} positive-rule share={cp.positive_effect_rule(pT, pU).mean():.3f}; top: "
          + ", ".join(f"{k}={v:.1%}" for k, v in top10[:5]) + f"; families {fam_share}")

    # link with the shipped probabilities the policy actually reads
    if "Proba_if_Treated" in rows_m.columns:
        sT, sU = rows_m["Proba_if_Treated"].to_numpy(), rows_m["Proba_if_Untreated"].to_numpy()
        out["link_to_state"] = {"corr_pT": float(np.corrcoef(pT, sT)[0, 1]), "corr_pU": float(np.corrcoef(pU, sU)[0, 1]),
                                "corr_cate": float(np.corrcoef(pU - pT, sU - sT)[0, 1]),
                                "positive_rule_agreement": float((cp.positive_effect_rule(pT, pU) == cp.positive_effect_rule(sT, sU)).mean())}
        print(f"  link: {out['link_to_state']}")

    ref = box.reference(X)
    out["deletion_test"] = {str(k): deletion_test_effect(box, X, phi, ref, k, seed=args.seed) for k in (1, 3)}
    print("  deletion:", {k: (round(v["z"], 1), round(v["flip_positive_effect_rule"], 3)) for k, v in out["deletion_test"].items()})

    # three-level cards on the paper's decision points
    ppo = PPO.load(str(paths.variant_model(name, args.variant)), device="cpu")
    margin = MarginHead(ppo.policy, intervene_action=1)
    wait_head = WaitMarginHead(margin)
    with torch.no_grad():
        m_all = margin(torch.from_numpy(states_m)).numpy()
    picks = {f"paper_card_{k}": v for k, v in pools.paper_card_indices(states_m, m_all, rows_m).items()}
    rclf, rfeats = rm.load_model(name)
    rbox = RiskBox(rclf, rfeats)
    import shap
    rex = shap.TreeExplainer(rclf)
    cards = {}
    for tag, i in picks.items():
        s = states_m[i:i + 1]
        dq = float(m_all[i]); acts = dq > 0
        phi_t = integrated_gradients(margin if acts else wait_head, s, states.mean(axis=0), n_steps=128)[0]
        rphi = np.asarray(rex.shap_values(X.iloc[[i]][rbox.columns]), dtype=np.float64)
        rphi = (rphi[..., 1] if rphi.ndim == 3 and rphi.shape[-1] == 2 else rphi).ravel()
        r = float(rbox.proba(X.iloc[[i]])[0])
        vals = {c: (X.iloc[i][c] if c in box.cat_cols else float(X.iloc[i][c])) for c in box.raw_columns}
        rvals = {c: (X.iloc[i][c] if c in rbox.cat_cols else float(X.iloc[i][c])) for c in rbox.columns}
        o = np.argsort(-np.abs(phi[i]))[:6]
        cards[tag] = {"case_id": str(meta.case_id.iloc[i]), "prefix_nr": int(meta.prefix_nr.iloc[i]), "action": "intervene" if acts else "wait",
                      "dq": dq, "r": r, "pT": float(pT[i]), "pU": float(pU[i]), "state": dict(zip(sfeats, s[0].astype(float).tolist())),
                      "effect_top": [(box.raw_columns[j], vals[box.raw_columns[j]], float(phi[i, j]), float(phiT[i, j]), float(phiU[i, j])) for j in o],
                      "effect_share_top6": float(np.abs(phi[i])[o].sum() / np.abs(phi[i]).sum()),
                      "timing_phi": dict(zip(sfeats, phi_t.astype(float).tolist()))}
        if not args.skip_plots:
            files += three_level_card(rphi, rbox.columns, rvals, r, phi[i], box.raw_columns, vals, float(pT[i]), float(pU[i]),
                                      phi_t, sfeats, dq, acts, figdir, f"{name}_{tag}_three_level_card")
        print(f"  card {tag}: {cards[tag]['case_id']} prefix {cards[tag]['prefix_nr']} {cards[tag]['action']} r={r:.3f} pT={pT[i]:.3f} pU={pU[i]:.3f} "
              f"effect top: {[(a, round(c, 2)) for a, _, c, _, _ in cards[tag]['effect_top'][:4]]}")
    out["cards"] = cards

    if not args.skip_plots:
        fig, ax = plt.subplots(figsize=(4.2, 2.6))
        ax.bar(list(fam_share), list(fam_share.values()), color=["#4c72b0", "#c44e52", "#2e8b57"])
        ax.set_ylabel("share of mean |phi^CATE|"); ax.set_title(f"{name}: effect attribution by family", fontsize=9)
        files += _save(fig, figdir, f"{name}_effect_family_share")
    out["figures"] = files
    out["seconds"] = time.perf_counter() - t0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", default=LOGS, choices=LOGS)
    ap.add_argument("--variant", default=pools.DEFAULT_VARIANT, choices=list(pools.VARIANTS))
    ap.add_argument("--seed", type=int, default=pools.POOL_SEED)
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    results = [run_log(name, args) for name in args.logs]
    out_path = Path(args.out) if args.out else (paths.EFFECT_JSON if args.variant == pools.DEFAULT_VARIANT
                                                 else paths.REPO / f"effect_results_{args.variant}.json")
    out_path.write_text(json.dumps(_jsonable({"args": vars(args), "results": results}), indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
