"""The composition study (paper Section 6): how the risk, effect and timing
explanations of RETIME compose on the evaluation pool, per BPIC log.

Per log this computes, on the Section 6 pool matched to the prepared log:

  1. the two readings of the state -- *shipped* (the RL CSV's numbers the
     checkpoint was trained on) and *rebuilt* (the same prefixes scored by the
     retrained risk and effect models) -- and how far they agree: per-level
     verdict agreement over the pool and the share of decisions that flip;
  2. for each reading: the timing justification phi^{Delta Q} on the side the
     policy chose, its level shares (Table 'card'), its propagation to the
     prefix attributes through phi^r, phi^{p_T}, phi^{p_U} (compose.propagate),
     the top attributes per channel and side, and the (risky) x (treatable) x
     (acts) typology with the level shares per cell;
  3. on the rebuilt reading, where the composed function F(x, n) = Delta Q(s(x,
     n)) exists: a direct Shapley-sampling attribution of F over the prefix
     attributes + natives, its agreement with the propagated one, and the
     deletion test of both rankings on F against random and anti-guided masking;
  4. the agreement of the risk and effect explanations on the shared prefix
     vocabulary;
  5. the paper's decision points (pools.paper_card_indices on the shipped
     margin, so the cases are those of Figs. 3 and 5) with their per-channel
     propagated attributions, for figures/make_composition_flow.py.

Writes paths.COMPOSE_JSON and figures under paths.COMPOSE_FIGURES/<log>/.

Usage: python run_compose_suite.py [--logs BPIC2017] [--quick 60] [--n-perm 20] [--skip-direct] [--skip-plots]
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
from stable_baselines3 import PPO  # noqa: E402

import compose as cp  # noqa: E402
import effect_model as em  # noqa: E402
import paths  # noqa: E402
import pools  # noqa: E402
import risk_model as rm  # noqa: E402
from run_effect_suite import EffectBox  # noqa: E402
from run_risk_suite import RiskBox, _save  # noqa: E402
from run_xai_suite import _jsonable  # noqa: E402

LOGS = ("BPIC2012", "BPIC2017", "SimBank", "Sepsis")
TOP = 8


def _risk_shap(clf, X: pd.DataFrame, columns: list[str]) -> np.ndarray:
    import shap

    ex = shap.TreeExplainer(clf)
    phi = np.asarray(ex.shap_values(X[columns]), dtype=np.float64)
    if phi.ndim == 3:
        phi = phi[..., 1] if phi.shape[-1] == 2 else phi[1]
    return phi


def _top(names: list[str], phi: np.ndarray, top: int = TOP) -> list[tuple[str, float]]:
    o = np.argsort(-np.abs(phi))[:top]
    return [(names[j], float(phi[j])) for j in o]


def _global_table(names: list[str], prop: dict, mask: np.ndarray, top: int = TOP, lower_attr: dict | None = None) -> dict:
    """Mean |composed phi| per input with the channel split and the
    cancellation index, on the states in ``mask``. With ``lower_attr``
    ({"risk": phi_r, "effect_T": phi^{p_T}, "effect_U": phi^{p_U}} over the
    prefix attributes) each non-native row also carries how the cancellation
    arises: the attribute's share of the risk explanation, the share of
    decisions where it moves both effect arms in the same direction, and the
    share where its two effect channels push the timing decision in opposite
    directions."""
    comp = prop["composed"][mask]
    g = np.abs(comp).mean(axis=0)
    share = g / g.sum()
    chan = {c: np.abs(prop["channels"][c][mask]).mean(axis=0) for c in ("risk", "effect_T", "effect_U")}
    sub = {"channels": {c: prop["channels"][c][mask] for c in ("risk", "effect_T", "effect_U")}}
    canc = cp.cancellation(sub)
    rows = []
    for j in np.argsort(-g)[:top]:
        name = names[j]
        d = {"input": name, "share": float(share[j]), "mean_abs": float(g[j]), "mean_signed": float(comp[:, j].mean()),
             "channel": {"native": float(g[j])} if name in cp.NATIVE else {c: float(chan[c][j]) for c in chan},
             "cancellation": None if name in cp.NATIVE else float(canc["per_attribute"][j])}
        if lower_attr is not None and name not in cp.NATIVE:
            fr, fT, fU = (lower_attr[c][mask] for c in ("risk", "effect_T", "effect_U"))
            gr = np.abs(fr).mean(axis=0)
            both = (fT[:, j] != 0) & (fU[:, j] != 0)
            cT, cU = prop["channels"]["effect_T"][mask][:, j], prop["channels"]["effect_U"][mask][:, j]
            both_ch = (cT != 0) & (cU != 0)
            d["risk_share"] = float(gr[j] / gr.sum()) if gr.sum() > 0 else 0.0
            d["arms_same_direction"] = float((np.sign(fT[both, j]) == np.sign(fU[both, j])).mean()) if both.any() else None
            d["channels_opposed"] = float((np.sign(cT[both_ch]) != np.sign(cU[both_ch])).mean()) if both_ch.any() else None
        rows.append(d)
    # share of the total channel mass (channels can cancel inside an input, so
    # the masses are normalised among themselves rather than by |composed|)
    masses = {c: float(np.abs(prop["channels"][c][mask]).sum()) for c in chan}
    masses["native"] = float(np.abs(prop["native"][mask]).sum())
    channel_share = {c: v / sum(masses.values()) for c, v in masses.items()}
    channel_share["cancellation"] = float(1.0 - np.abs(comp).sum() / sum(masses.values()))  # mass lost to opposite-sign channels
    n_native = prop["native"].shape[1]  # 1 on Sepsis (no available_resources), 2 elsewhere
    by_arms = None
    if lower_attr is not None:
        # weight lost to cancelling over every (decision, attribute) pair, split by whether the
        # attribute moves both effect arms in the same direction (the risk-like case) or not
        fT, fU = lower_attr["effect_T"][mask], lower_attr["effect_U"][mask]
        chs = [prop["channels"][c][mask] for c in ("risk", "effect_T", "effect_U")]
        net, mass = np.abs(sum(chs)), sum(np.abs(c) for c in chs)
        both = (fT != 0) & (fU != 0)
        same = both & (np.sign(fT) == np.sign(fU))
        diff = both & (np.sign(fT) != np.sign(fU))
        lost = lambda m: float(1 - net[m].sum() / mass[m].sum()) if mass[m].sum() > 0 else None
        by_arms = {"lost_same_direction": lost(same), "lost_opposite_direction": lost(diff),
                   "mass_share_same_direction": float(mass[same].sum() / mass[both].sum()) if mass[both].sum() > 0 else None}
    return {"n": int(mask.sum()), "top": rows, "cancellation_by_arm_direction": by_arms, "channel_share": channel_share, "cancellation_overall": canc["overall"],
            "cancellation_per_state_mean": float(canc["per_state"].mean()),
            "prefix_share_top5": float(np.sort(share[:len(share) - n_native])[::-1][:5].sum())}


def _flows(i: int, names: list[str], prop: dict, xrow, top: int = TOP) -> list[dict]:
    """Per top input of state i: its value and its contribution through every channel."""
    comp = prop["composed"][i]
    n_native = prop["native"].shape[1]
    n_raw = len(names) - n_native

    def val(name):
        v = xrow[name]
        return v if isinstance(v, str) else float(v)

    o = np.argsort(-np.abs(comp[:n_raw]))[:top]
    flows = [{"input": names[j], "value": val(names[j]), "risk": float(prop["channels"]["risk"][i, j]), "effect_T": float(prop["channels"]["effect_T"][i, j]),
              "effect_U": float(prop["channels"]["effect_U"][i, j]), "total": float(comp[j])} for j in o]
    rest = np.setdiff1d(np.arange(n_raw), o)
    flows.append({"input": "(other attributes)", "risk": float(prop["channels"]["risk"][i, rest].sum()),
                  "effect_T": float(prop["channels"]["effect_T"][i, rest].sum()), "effect_U": float(prop["channels"]["effect_U"][i, rest].sum()),
                  "total": float(comp[rest].sum())})
    return flows


def _card(i: int, names: list[str], prop: dict, phi_t: np.ndarray, state: np.ndarray, meta_row, xrow, r, pT, pU, dq, feats: list[str], top: int = TOP) -> dict:
    n_native = prop["native"].shape[1]
    n_raw = len(names) - n_native
    native_names = [f for f in cp.NATIVE if f in feats]  # order matches compose.propagate's native_feats
    per_chan = {}
    for c in ("risk", "effect_T", "effect_U"):
        v = prop["channels"][c][i]
        per_chan[c] = _top(names[:n_raw], v, top)
    comp = prop["composed"][i]
    flows = _flows(i, names, prop, xrow, top)
    return {"case_id": str(meta_row.case_id), "prefix_nr": int(meta_row.prefix_nr), "action": "intervene" if dq > 0 else "wait", "dq": float(dq),
            "r": float(r), "pT": float(pT), "pU": float(pU), "state": dict(zip(feats, state.astype(float).tolist())),
            "timing_phi": dict(zip(feats, phi_t.astype(float).tolist())), "native_phi": dict(zip(native_names, prop["native"][i].tolist())),
            "composed_top": _top(names, comp, top), "per_channel_top": per_chan, "flows": flows,
            "completeness": {"timing_sum": float(phi_t.sum()), "composed_sum": float(comp.sum())}}


def _plot_top(names, prop, mask, title, out: Path, stem: str, top: int = 10) -> list[str]:
    comp = prop["composed"][mask]
    g = np.abs(comp).mean(axis=0)
    o = np.argsort(-g)[:top]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bottoms = np.zeros(len(o))
    colors = {"risk": "#4c72b0", "effect_T": "#c44e52", "effect_U": "#dd8452", "native": "#55a868"}
    for c in ("risk", "effect_T", "effect_U"):
        v = np.array([np.abs(prop["channels"][c][mask][:, j]).mean() if names[j] not in cp.NATIVE else 0.0 for j in o])
        ax.barh(range(len(o)), v, left=bottoms, color=colors[c], label=c)
        bottoms += v
    v = np.array([g[j] if names[j] in cp.NATIVE else 0.0 for j in o])
    ax.barh(range(len(o)), v, left=bottoms, color=colors["native"], label="native")
    ax.set_yticks(range(len(o)), [names[j] for j in o], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("mean |composed phi| (units of the margin)", fontsize=8); ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return _save(fig, out, stem)


def run_log(name: str, args) -> dict:
    t0 = time.perf_counter()
    states, rows, sfeats = pools.evaluation_pool(name, args.variant)
    has_effect = "Proba_if_Treated" in sfeats
    clf, rfeats = rm.load_model(name)
    rbox = RiskBox(clf, rfeats)
    arms, efeats = em.load_model(name)
    ebox = EffectBox(arms, efeats)
    ppo = PPO.load(str(paths.variant_model(name, args.variant)), device="cpu")
    box = cp.EndToEndBox(name, rbox, ebox, ppo.policy, feats=sfeats)
    names = box.columns

    X, meta = rm.prefixes_for_pool(name, rows)
    keep = meta["pool_row"].to_numpy()
    states_m, rows_m = states[keep], rows.iloc[keep].reset_index(drop=True)
    if args.quick and len(X) > args.quick:
        sel = np.random.default_rng(pools.POOL_SEED).choice(len(X), args.quick, replace=False)
        X, meta, states_m, rows_m = X.iloc[sel].reset_index(drop=True), meta.iloc[sel].reset_index(drop=True), states_m[sel], rows_m.iloc[sel].reset_index(drop=True)
    n = len(X)
    print(f"\n=== {name}: {n} pool rows matched; {len(box.raw_columns)} prefix attributes + {len(box.native)} native")
    out: dict = {"log": name, "n": int(n), "inputs": names, "state_features": sfeats, "variant": args.variant}
    figdir = paths.COMPOSE_FIGURES / name
    files: list[str] = []

    # --- lower levels on the pool ----------------------------------------
    r = rbox.proba(X)
    pT, pU = ebox.probs(X)
    phi_r = _risk_shap(clf, X, rbox.columns)  # CatBoost: tree-path-dependent only (categorical splits)
    bg = None if args.background == "tree" else X.sample(n=min(args.n_background, len(X)), random_state=args.seed)
    phiT, phiU, ev = ebox.shap_raw(X, background=bg)
    lower = {f: v for f, v in {"reliability": phi_r, "deviation": phi_r, "Proba_if_Treated": phiT, "Proba_if_Untreated": phiU}.items() if f in sfeats}
    lower_attr = {"risk": phi_r, "effect_T": phiT, "effect_U": phiU}
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    out["well_defined"] = cp.well_defined({"risk (r)": phi_r, "Proba_if_Treated": phiT, "Proba_if_Untreated": phiU})
    out["baseline"] = {"background": args.background, "n_background": None if bg is None else int(len(bg)),
                       "risk": {"explainer": "tree_path_dependent", **cp.baseline_alignment(phi_r, lg(r))},
                       "Proba_if_Treated": {"explainer": "interventional(pool)" if bg is not None else "tree_path_dependent", **cp.baseline_alignment(phiT, lg(pT))},
                       "Proba_if_Untreated": {"explainer": "interventional(pool)" if bg is not None else "tree_path_dependent", **cp.baseline_alignment(phiU, lg(pU))}}
    print("  well-defined:", {k: (round(v["share_defined"], 3) if isinstance(v, dict) else v) for k, v in out["well_defined"].items()})
    print("  baseline:", {k: {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("explainer", "baseline_gap", "sign_agreement", "corr")} for k, v in out["baseline"].items() if isinstance(v, dict)})

    # --- the two readings of the state ------------------------------------
    rebuilt = cp.rebuild_state(name, states_m, r, pT, pU, sfeats)
    readings = {"shipped": states_m, "rebuilt": rebuilt}
    m_ship, m_reb = cp.margin_of(ppo.policy, states_m), cp.margin_of(ppo.policy, rebuilt)
    ship = pd.DataFrame(states_m, columns=sfeats)
    shipped_r = rows_m["predicted_proba_1"].to_numpy(float) if "predicted_proba_1" in rows_m else None
    oracle = pools.oracle_acts(rows_m)
    rule_re = cp.positive_effect_rule(pT, pU)
    idev, irel = sfeats.index("deviation"), sfeats.index("reliability")
    out["state_agreement"] = {
        "risk": {"deviation_agrees": float((rebuilt[:, idev] == ship.deviation.to_numpy()).mean()),
                 "predicted_deviant_shipped": float((ship.deviation == 0).mean()), "predicted_deviant_rebuilt": float((rebuilt[:, idev] == 0).mean()),
                 "corr_r_shipped": float(np.corrcoef(r, shipped_r)[0, 1]) if shipped_r is not None else None,
                 "reliability_mean_abs_diff": float(np.abs(rebuilt[:, irel] - ship.reliability.to_numpy()).mean())},
        "effect": {"positive_rule_agrees": float((rule_re == oracle).mean()), "positive_rule_shipped": float(oracle.mean()), "positive_rule_rebuilt": float(rule_re.mean()),
                   "corr_pT": float(np.corrcoef(pT, ship.Proba_if_Treated)[0, 1]) if has_effect else None,
                   "corr_pU": float(np.corrcoef(pU, ship.Proba_if_Untreated)[0, 1]) if has_effect and ship.Proba_if_Untreated.std() > 0 else None,
                   "mean_abs_diff_pT": float(np.abs(pT - ship.Proba_if_Treated).mean()) if has_effect else None,
                   "mean_abs_diff_pU": float(np.abs(pU - ship.Proba_if_Untreated).mean()) if has_effect else None},
        "decision": {"intervene_rate_shipped": float((m_ship > 0).mean()), "intervene_rate_rebuilt": float((m_reb > 0).mean()),
                     "flip_rate": float(((m_ship > 0) != (m_reb > 0)).mean()), "corr_margin": float(np.corrcoef(m_ship, m_reb)[0, 1]),
                     "agreement_with_oracle_shipped": float(((m_ship > 0) == oracle).mean()), "agreement_with_oracle_rebuilt": float(((m_reb > 0) == oracle).mean()),
                     "agreement_with_rebuilt_rule_rebuilt": float(((m_reb > 0) == rule_re).mean())},
    }
    print("  state agreement:", json.dumps(out["state_agreement"]["decision"]))
    print("    risk:", json.dumps(out["state_agreement"]["risk"]), "\n    effect:", json.dumps(out["state_agreement"]["effect"]))

    # --- per reading: timing, shares, propagation, typology ----------------
    picks = pools.paper_card_indices(states_m, m_ship, rows_m)  # the paper's cases, fixed by the shipped reading
    out["readings"] = {}
    props = {}
    for reading, S in readings.items():
        ref = S.mean(axis=0)
        phi_t, sign = cp.timing_attribution(ppo.policy, S, ref)
        m = cp.margin_of(ppo.policy, S)
        acts = m > 0
        prop = cp.propagate(phi_t, lower, sfeats)
        props[reading] = (phi_t, sign, prop, m)
        risky = (rebuilt[:, idev] == 0) if reading == "rebuilt" else (ship.deviation.to_numpy() == 0)
        treatable = rule_re if reading == "rebuilt" else oracle
        rd = {"reference": dict(zip(sfeats, ref.astype(float).tolist())), "n_act": int(acts.sum()), "n_wait": int((~acts).sum()),
              "level_shares": {"act": cp.level_shares(phi_t[acts], sfeats) if acts.any() else None, "wait": cp.level_shares(phi_t[~acts], sfeats) if (~acts).any() else None,
                               "all": cp.level_shares(phi_t, sfeats)},
              "propagation": {"completeness_gap": cp.completeness_gap(phi_t, prop["composed"]), "fallbacks": prop["fallbacks"],
                              "act": _global_table(names, prop, acts, lower_attr=lower_attr) if acts.any() else None,
                              "wait": _global_table(names, prop, ~acts, lower_attr=lower_attr) if (~acts).any() else None},
              "typology": cp.typology(risky, treatable, acts, phi_t),
              "cards": {}}
        for tag, i in picks.items():
            rd["cards"][f"paper_card_{tag}"] = _card(i, names, prop, phi_t[i], S[i], meta.iloc[i], X.iloc[i], r[i], pT[i], pU[i], m[i], sfeats)
        out["readings"][reading] = rd
        ls = rd["level_shares"]
        print(f"  [{reading}] act={acts.sum()} wait={(~acts).sum()} level shares act={ {k: round(v, 3) for k, v in (ls['act'] or {'per_level': {}})['per_level'].items()} } "
              f"wait={ {k: round(v, 3) for k, v in (ls['wait'] or {'per_level': {}})['per_level'].items()} } fallbacks={prop['fallbacks']}")
        for side in ("act", "wait"):
            t = rd["propagation"][side]
            if t:
                print(f"    composed top ({side}): " + ", ".join(f"{d['input']}={d['share']:.1%}" for d in t["top"][:6]) + f"; channels {{{', '.join(f'{k}={v:.2f}' for k, v in t['channel_share'].items())}}}")
        print("    typology:", {k: v["n"] for k, v in rd["typology"].items() if v["n"]})
        if not args.skip_plots:
            for side, mask in (("act", acts), ("wait", ~acts)):
                if mask.any():
                    files += _plot_top(names, prop, mask, f"{name} ({reading} state), {side} side: what the timing level inherits", figdir, f"{name}_{reading}_{side}_composed_top")

    # --- direct attribution of the composed function (rebuilt reading) ------
    phi_t, sign, prop, m = props["rebuilt"]
    Z = box.inputs(X, rebuilt)
    zref = box.reference(Z)
    assert np.allclose(box.f(Z), m, atol=1e-4)
    rankings = {"propagated": prop["composed"]}
    if not args.skip_direct:
        t1 = time.perf_counter()
        zbg = None if args.background == "tree" else box.inputs(bg, states_m[bg.index.to_numpy()])  # X has a RangeIndex: positions = labels
        phi_d = cp.shapley_sampling(box, Z, zref, sign, n_perm=args.n_perm, seed=args.seed, background=zbg)
        f_ref = (sign * np.mean(box.f(zbg)) if zbg is not None else box.f(pd.DataFrame([zref] * n).reset_index(drop=True), sign))
        rankings["direct"] = phi_d
        # the per-decision (direct-anchored) composition: direct ranking, propagated channel split
        anch = cp.anchor_to_direct(phi_d, prop)
        rd = out["readings"]["rebuilt"]
        rd["anchored"] = {"act": _global_table(names, anch, m > 0) if (m > 0).any() else None,
                          "wait": _global_table(names, anch, m <= 0) if (m <= 0).any() else None}
        for tag, i in picks.items():
            card = rd["cards"][f"paper_card_{tag}"]
            card["anchored_flows"] = _flows(i, names, anch, X.iloc[i])
            card["anchored_top"] = _top(names, anch["composed"][i], TOP)
            card["direct_top"] = _top(names, phi_d[i], TOP)
            card["direct_vs_propagated"] = {"spearman": float(cp.per_state_spearman(prop["composed"][i:i + 1], phi_d[i:i + 1])[0]),
                                            "same_top": bool(np.argmax(np.abs(prop["composed"][i])) == np.argmax(np.abs(phi_d[i])))}
        rho_s = cp.per_state_spearman(prop["composed"], phi_d)
        canc_s = cp.cancellation(prop)["per_state"]
        from scipy.stats import spearmanr as _sp
        okc = np.isfinite(rho_s)
        med = float(np.median(canc_s[okc]))
        agree_canc = {"spearman_rho_vs_cancellation": float(_sp(rho_s[okc], canc_s[okc])[0]), "median_cancellation": med,
                      "agreement_low_cancellation": float(rho_s[okc & (canc_s <= med)].mean()), "agreement_high_cancellation": float(rho_s[okc & (canc_s > med)].mean()),
                      "n_low": int((okc & (canc_s <= med)).sum()), "n_high": int((okc & (canc_s > med)).sum())}
        print("  agreement vs cancellation:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in agree_canc.items()})
        out["direct"] = {"n_perm": args.n_perm, "seconds": time.perf_counter() - t1, "agreement_vs_cancellation": agree_canc,
                         "reference": "pool background" if zbg is not None else "point (pool mean/mode)",
                         "F_reference_mean": float(np.mean(box.f(zbg))) if zbg is not None else float(box.f(pd.DataFrame([zref]))[0]),
                         "completeness_gap": float(np.abs(phi_d.sum(axis=1) - (box.f(Z, sign) - f_ref)).max()),
                         # sampled Shapley values sum to f(Z) minus the mean of *their own* background draws, so
                         # against the pool expectation they are complete only on average: report the typical gap too
                         "completeness_gap_mean": float(np.abs(phi_d.sum(axis=1) - (box.f(Z, sign) - f_ref)).mean()),
                         "completeness_gap_signed_mean": float((phi_d.sum(axis=1) - (box.f(Z, sign) - f_ref)).mean()),
                         "margin_abs_mean": float(np.abs(box.f(Z, sign) - f_ref).mean()),
                         "agreement_with_propagated": {"all": cp.ranking_agreement(prop["composed"], phi_d),
                                                       "act": cp.ranking_agreement(prop["composed"][m > 0], phi_d[m > 0]) if (m > 0).any() else None,
                                                       "wait": cp.ranking_agreement(prop["composed"][m <= 0], phi_d[m <= 0]) if (m <= 0).any() else None},
                         "global_top": {side: _top(names, np.abs(phi_d[mask]).mean(axis=0), TOP) for side, mask in (("act", m > 0), ("wait", m <= 0)) if mask.any()},
                         "native_share_direct": {side: float(np.abs(phi_d[mask][:, -len(box.native):]).sum() / np.abs(phi_d[mask]).sum())
                                                 for side, mask in (("act", m > 0), ("wait", m <= 0)) if mask.any()}}
        print(f"  direct Shapley sampling ({args.n_perm} perms, {out['direct']['seconds']:.0f}s): agreement", json.dumps({k: round(v, 3) for k, v in out["direct"]["agreement_with_propagated"]["all"].items()}))
    out["deletion_e2e"] = {}
    for side, mask in (("act", m > 0), ("wait", m <= 0)):
        if mask.sum() < 30:
            continue
        Zs = Z[mask].reset_index(drop=True)
        out["deletion_e2e"][side] = {str(k): cp.deletion_test_e2e(box, Zs, zref, sign[mask], {nm: v[mask] for nm, v in rankings.items()}, k, seed=args.seed) for k in (1, 3)}
        for k, res in out["deletion_e2e"][side].items():
            print(f"  deletion e2e [{side}] k={k}: random={res['abs_random']:.3f} " + " ".join(f"{nm}: guided={res[nm]['abs_guided']:.3f} anti={res[nm]['abs_anti']:.3f} z={res[nm]['z']:.1f}" for nm in rankings))

    # --- do the channels and the cancellation hold on the real chain? ---------
    if has_effect:
        out["channel_test"] = {}
        for side, mask in (("act", m > 0), ("wait", m <= 0)):
            if mask.sum() < 30:
                continue
            sub = {"composed": prop["composed"][mask], "channels": {c: prop["channels"][c][mask] for c in ("risk", "effect_T", "effect_U")}}
            out["channel_test"][side] = cp.channel_test(box, Z[mask].reset_index(drop=True), zref, sign[mask], sub,
                                                        {c: v[mask] for c, v in lower_attr.items()})
            ct = out["channel_test"][side]
            print(f"  channel test [{side}]: " + ", ".join(f"{c} sign={v['sign_agreement']} rho={v['spearman']} (n={v['n']})" for c, v in ct["channels"].items())
                  + f"; cancellation {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in ct['cancellation'].items()})}")

    # --- risk vs effect on the shared vocabulary ----------------------------
    out["risk_effect"] = cp.risk_effect_agreement(phi_r, phiU - phiT, box.raw_columns)  # phi^CATE: toward a larger benefit
    out["risk_effect"]["groups"] = cp.sign_groups(out["risk_effect"]["per_attribute"])
    print("  sign groups:", {k: v for k, v in out["risk_effect"]["groups"].items()})
    print(f"  risk vs effect: global rho={out['risk_effect']['global_spearman']:.2f} jaccard@10={out['risk_effect']['jaccard_top10']:.2f} shared={out['risk_effect']['shared_top']}")

    out["figures"] = files
    out["seconds"] = time.perf_counter() - t0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", default=list(LOGS), choices=list(LOGS))
    ap.add_argument("--variant", default=pools.DEFAULT_VARIANT, choices=list(pools.VARIANTS))
    ap.add_argument("--seed", type=int, default=pools.POOL_SEED)
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--background", choices=["pool", "tree"], default="pool",
                    help="background of the effect arms' TreeSHAP: a sample of the pool's prefixes (interventional; aligns the "
                         "lower-level baseline with the timing level's pool reference) or the tree's own path-dependent default")
    ap.add_argument("--n-background", type=int, default=100)
    ap.add_argument("--skip-direct", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--fallback-ratio", type=float, default=cp.FALLBACK_RATIO,
                    help="stabiliser of the proportional weights; 0 divides as Chen et al. (2022) do (robustness run)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cp.FALLBACK_RATIO = args.fallback_ratio
    results = [run_log(name, args) for name in args.logs]
    out_path = Path(args.out) if args.out else (paths.COMPOSE_JSON if args.variant == pools.DEFAULT_VARIANT
                                                 else paths.REPO / f"compose_results_{args.variant}.json")
    out_path.write_text(json.dumps(_jsonable({"args": vars(args), "results": results}), indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
