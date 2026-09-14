"""Write the manuscript's numbers as LaTeX macros (numbers.tex) from the
committed result JSONs, so every figure quoted in the paper is regenerated
from the same files the tables are drawn from.

Reads fidelity_results.json, gain_table_results.json, risk_results.json,
effect_results.json, compose_results.json (the paper's policy) and
compose_results_cate.json (the shipped-state design, for the drift numbers),
plus the retrained-state CSV and the policy manifest, and writes
``\\newcommand`` definitions and table bodies.

Usage: TIMING_PAPER_FIGURES=<paper>/figures python paper_numbers.py   # -> <paper>/numbers.tex (next to figures/)
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import compose as cp
import paths
import pools

LOG = "BPIC2017"
NAMES = {"relative_position": r"\texttt{relative\_position}", "reliability": r"\texttt{reliability}", "deviation": r"\texttt{deviation}",
         "available_resources": r"\texttt{available\_resources}", "Proba_if_Treated": "Treated-arm probability",
         "Proba_if_Untreated": "Untreated-arm probability"}
LEVEL_NAME = {"native": "Native", "risk": "Risk", "effect": "Effect"}
CHAN_NAME = {"risk": "risk", "effect_T": "effect (treated arm)", "effect_U": "effect (untreated arm)", "native": "native"}


def tex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def num(x, d=2):
    return f"{x:.{d}f}"


def macro(name: str, value) -> str:
    words = {"0": "ZERO", "1": "ONE", "2": "TWO", "3": "THREE", "5": "FIVE"}
    name = "".join(words.get(ch, ch) for ch in name)
    name = re.sub(r"[^A-Za-z]", "", name)
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def main():
    out = []
    # SimBank is excluded from these companion numbers: its `cate` checkpoint's
    # reliability/deviation come from the simulator's own est_quality/unc_quality
    # (train_ppo_simbank.py), not from a retrainable risk classifier r(x) as on
    # BPIC, so compose.py's generic rebuild_state does not describe what that
    # policy actually reads (shipped-vs-rebuilt deviation_agrees ~= 0.40, not
    # ~1.0). Reporting it here would be reporting an incoherent pipeline. Fixing
    # this needs a redesigned SimBank environment (risk features from a retrained
    # classifier) and a PPO agent retrained on it -- not attempted.
    all_fid = {x["log"]: x for x in json.loads(paths.FIDELITY_JSON.read_text())}
    fid = all_fid[LOG]
    fid12 = all_fid.get("BPIC2012")
    fid_sep = all_fid.get("Sepsis")
    all_comp = {x["log"]: x for x in json.loads(paths.COMPOSE_JSON.read_text())["results"]}
    comp = all_comp[LOG]
    comp12 = all_comp.get("BPIC2012")
    comp_sep = all_comp.get("Sepsis")
    comp_cate_path = paths.REPO / "compose_results_cate.json"
    comp_cate = {x["log"]: x for x in json.loads(comp_cate_path.read_text())["results"]} if comp_cate_path.exists() else None
    risk = {x["log"]: x for x in json.loads(paths.RISK_JSON.read_text())["results"]}[LOG]
    eff = {x["log"]: x for x in json.loads(paths.EFFECT_JSON.read_text())["results"]}[LOG]
    gain = json.loads(paths.GAIN_JSON.read_text())["bpic2017"]
    manifest = json.loads(paths.variant_artifact(LOG, pools.DEFAULT_VARIANT, "_manifest.json").read_text())
    csv = pd.read_csv(paths.bpic_csv(LOG, pools.DEFAULT_VARIANT), sep=";", usecols=["case_id", "deviation", "y1", "y0"])

    # --- Section 5 -----------------------------------------------------------
    out += [macro("NTESTPREFIXES", f"{len(csv):,}"), macro("NTESTCASES", f"{csv.case_id.nunique():,}"),
            macro("PREDDEVIANT", pct((csv.deviation == 0).mean(), 0)), macro("ITEPOS", pct((csv.y1 - csv.y0 > 0).mean(), 0)),
            macro("INTERVENERATE", pct(gain["policy_intervene_rate"], 1)), macro("PPOSTEPS", f"{manifest['total_timesteps']:,}"),
            macro("GAINPOLICY", num(gain["policy_gain"], 1)), macro("GAINWAIT", num(gain["always_wait_gain"], 1)), macro("GAINORACLE", num(gain["oracle_gain"], 1)),
            macro("GAINHIST", num(gain["hist_gain"], 1)), macro("POLICYPRECISION", num(gain["policy_precision"], 2)), macro("POLICYRECALL", num(gain["policy_recall"], 2))]

    # --- Section 5 materials: the two lower-level models and the agent's recipe ---
    risk_m = json.loads((paths.RISK_MODELS / f"{LOG}_manifest.json").read_text())
    eff_m = json.loads((paths.EFFECT_MODELS / f"{LOG}_manifest.json").read_text())
    out += [macro("NTRAINPREFIXES", f"{risk_m['n_train']:,}"), macro("NVALPREFIXES", f"{risk_m['n_val']:,}"),
            macro("RISKAUC", num(risk_m["test_auc"], 2)), macro("RISKITERATIONS", f"{risk_m['iterations']:,}"), macro("RISKBESTITER", str(risk_m["best_iteration"])),
            macro("EFFECTAUCT", num(eff_m["auc_treated_arm_on_treated_test"], 2)), macro("EFFECTAUCU", num(eff_m["auc_untreated_arm_on_untreated_test"], 2)),
            macro("NEFFECTFEATURES", str(eff_m["n_features"])), macro("TREATEDSHARE", pct(eff_m["treated_share_train"], 0)),
            macro("PPOSEED", str(manifest["seed"])), macro("PPOENTCOEF", str(manifest["ent_coef"])), macro("PPOREWARDSCALE", str(manifest["reward_scale"])),
            macro("PPOLR", str(manifest["learning_rate"])), macro("PPONSTEPS", str(manifest["n_steps"])), macro("PPOEPISODES", f"{manifest['n_episodes']:,}"),
            macro("PPORESOURCES", str(manifest["args"]["resources"]))]
    half_path = paths.variant_artifact(LOG, pools.DEFAULT_VARIANT + "_300k", "_manifest.json")
    if half_path.exists():
        half = json.loads(half_path.read_text())
        m = re.search(r"\((\d+\.\d+)%, precision (\d\.\d+), gain (\d+\.\d+)", half.get("note", ""))
        out.append(macro("HALFSTEPS", f"{half['total_timesteps']:,}"))
        if m:
            out += [macro("HALFINTERVENERATE", m.group(1) + r"\%"), macro("HALFPRECISION", m.group(2)), macro("HALFGAIN", m.group(3))]

    # --- Section 6.0 pool ------------------------------------------------------
    n_act, n_wait = fid["n_intervene_states"], fid["n_wait_states"]
    ref = dict(zip(fid["feature_names"], fid["reference"]))
    out += [macro("NACT", str(n_act)), macro("NWAIT", str(n_wait)), macro("NPOOL", str(fid["n_pool_states"])),
            ]
    ref_margin = float(cp.margin_of(__import__("stable_baselines3").PPO.load(str(paths.variant_model(LOG, pools.DEFAULT_VARIANT)), device="cpu").policy,
                                    np.array([fid["reference"]], np.float32))[0])
    out += [macro("REFMARGIN", num(ref_margin, 2)), macro("REFWAITMARGIN", num(-ref_margin, 2)),
            macro("REFACTION", "acts" if ref_margin > 0 else "waits")]
    if fid12:
        out += [macro("NACTBPIC", str(fid12["n_intervene_states"])), macro("NWAITBPIC", str(fid12["n_wait_states"]))]

    # --- Table card: level shares (BPIC2012 in brackets alongside each cell) -----
    sh = {side: fid[key]["phi_share"] for side, key in (("act", "intervene_side"), ("wait", "wait_side")) if key in fid}
    feats = fid["feature_names"]
    sh12 = ({side: fid12[key]["phi_share"] for side, key in (("act", "intervene_side"), ("wait", "wait_side")) if key in fid12}
            if fid12 else {})
    feats12 = fid12["feature_names"] if fid12 else []

    def cell(s, i, i12):
        v = f"${pct(sh[s][i])}$"
        if s in sh12 and i12 is not None:
            v += f" [${pct(sh12[s][i12])}$]"
        return v

    rows = []
    for level, fs in (("effect", ["Proba_if_Treated", "Proba_if_Untreated"]), ("risk", ["reliability", "deviation"]), ("native", ["available_resources", "relative_position"])):
        tot = {s: 0.0 for s in sh}
        tot12 = {s: 0.0 for s in sh12}
        for f in fs:
            i = feats.index(f)
            i12 = feats12.index(f) if f in feats12 else None
            rows.append(f"{NAMES[f]} & {LEVEL_NAME[level]} & " + " & ".join(cell(s, i, i12) for s in ("act", "wait") if s in sh) + r" \\ \hline")
            for s in sh:
                tot[s] += sh[s][i]
            for s in sh12:
                if i12 is not None:
                    tot12[s] += sh12[s][i12]
        tot_cell = lambda s: f"$\\mathbf{{{pct(tot[s])}}}$" + (f" [$\\mathbf{{{pct(tot12[s])}}}$]" if s in tot12 else "")
        rows.append(r"\rowcolor{keyrow} \textbf{" + LEVEL_NAME[level] + " level total} & & " + " & ".join(tot_cell(s) for s in ("act", "wait") if s in sh) + r" \\ \hline")
        for s in sh:
            out.append(macro(f"SHARE{level.upper()}{s.upper()}", pct(tot[s])))
        for s in sh12:
            out.append(macro(f"SHARE{level.upper()}{s.upper()}BPIC", pct(tot12[s])))
    out.append(macro("CARDROWS", "\n".join(rows)))
    for f in feats:
        i = feats.index(f)
        for s in sh:
            out.append(macro(f"SHARE{f.replace('_', '').replace('Probaif', 'P').upper()}{s.upper()}", pct(sh[s][i])))
    for f in feats12:
        i12 = feats12.index(f)
        for s in sh12:
            out.append(macro(f"SHARE{f.replace('_', '').replace('Probaif', 'P').upper()}{s.upper()}BPIC", pct(sh12[s][i12])))

    # --- Sepsis level shares (reported in prose: 5-feature state, own vocabulary) --
    if fid_sep:
        sh_sep = {side: fid_sep[key]["phi_share"] for side, key in (("act", "intervene_side"), ("wait", "wait_side")) if key in fid_sep}
        feats_sep = fid_sep["feature_names"]
        out += [macro("NACTSEPSIS", str(fid_sep["n_intervene_states"])), macro("NWAITSEPSIS", str(fid_sep["n_wait_states"]))]
        for level, fs in (("effect", ["Proba_if_Treated", "Proba_if_Untreated"]), ("risk", ["reliability", "deviation"]), ("native", ["relative_position"])):
            tot_sep = {s: 0.0 for s in sh_sep}
            for f in fs:
                if f not in feats_sep:
                    continue
                i = feats_sep.index(f)
                for s in sh_sep:
                    tot_sep[s] += sh_sep[s][i]
            for s in sh_sep:
                out.append(macro(f"SHARE{level.upper()}{s.upper()}SEPSIS", pct(tot_sep[s])))

    # --- Section 6.3: what the timing level inherits (propagated) ---------------
    rd = comp["readings"]["rebuilt"]
    inputs = comp["inputs"]
    for side in ("act", "wait"):
        t = rd["propagation"][side]
        rows = []
        for d in t["top"][:6]:
            ch = d["channel"]
            if "native" in ch:
                split = "native"
            else:
                tot = sum(ch.values()) or 1.0
                split = ", ".join(f"{CHAN_NAME[c]} {pct(v / tot, 0)}" for c, v in sorted(ch.items(), key=lambda kv: -kv[1]) if v / tot >= 0.05)
            canc = "--" if d.get("cancellation") is None else pct(d["cancellation"], 0)
            rows.append(f"\\texttt{{{tex(d['input'])}}} & ${pct(d['share'])}$ & {split} & {canc} \\\\ \\hline")
        out.append(macro(f"INHERIT{side.upper()}ROWS", "\n".join(rows)))
        out.append(macro(f"CANCEL{side.upper()}", pct(t["cancellation_overall"], 0)))
        out.append(macro(f"TOPINPUTCANC{side.upper()}", pct(t["top"][0]["cancellation"], 0) if t["top"][0].get("cancellation") is not None else "--"))
        top_c = [(d["input"], d["cancellation"]) for d in t["top"] if d.get("cancellation") is not None]
        hi = max(top_c, key=lambda kv: kv[1]); lo = min(top_c, key=lambda kv: kv[1])
        out += [macro(f"CANCELHI{side.upper()}", f"\\texttt{{{tex(hi[0])}}}"), macro(f"CANCELHIVAL{side.upper()}", pct(hi[1], 0)),
                macro(f"CANCELLO{side.upper()}", f"\\texttt{{{tex(lo[0])}}}"), macro(f"CANCELLOVAL{side.upper()}", pct(lo[1], 0))]
        cs = t["channel_share"]
        out += [macro(f"CHANRISK{side.upper()}", pct(cs["risk"])), macro(f"CHANEFFECTT{side.upper()}", pct(cs["effect_T"])),
                macro(f"CHANEFFECTU{side.upper()}", pct(cs["effect_U"])), macro(f"CHANEFFECT{side.upper()}", pct(cs["effect_T"] + cs["effect_U"])),
                macro(f"CHANNATIVE{side.upper()}", pct(cs["native"])), macro(f"TOPFIVE{side.upper()}", pct(t["prefix_share_top5"], 0)),
                macro(f"TOPINPUT{side.upper()}", f"\\texttt{{{tex(t['top'][0]['input'])}}}"), macro(f"TOPINPUTSHARE{side.upper()}", pct(t["top"][0]["share"], 0))]
    out += [macro("COMPLETENESSGAP", f"{rd['propagation']['completeness_gap']:.0e}".replace("e-0", "e-")),
            macro("FALLBACKS", str(sum(rd["propagation"]["fallbacks"].values()))), macro("NPREFIXATTRS", str(len(inputs) - 2))]

    wd = comp["well_defined"]
    out += [macro("WDRATIO", f"{wd['ratio']:.2f}"), macro("WDALL", pct(wd["share_all_defined"], 0)),
            macro("WDRISK", pct(wd["risk (r)"]["share_defined"], 0)), macro("WDPT", pct(wd["Proba_if_Treated"]["share_defined"], 0)), macro("WDPU", pct(wd["Proba_if_Untreated"]["share_defined"], 0))]
    if comp12 and "well_defined" in comp12:
        out.append(macro("WDALLBPIC", pct(comp12["well_defined"]["share_all_defined"], 0)))
        out.append(macro("WDRISKBPIC", pct(comp12["well_defined"]["risk (r)"]["share_defined"], 0)))
    if comp12 and "readings" in comp12:
        rd12 = comp12["readings"]["rebuilt"]
        for side in ("act", "wait"):
            if side in rd12["propagation"]:
                t12 = rd12["propagation"][side]
                cs12 = t12["channel_share"]
                out += [macro(f"TOPFIVE{side.upper()}BPIC", pct(t12["prefix_share_top5"], 0)),
                        macro(f"CHANRISK{side.upper()}BPIC", pct(cs12["risk"])),
                        macro(f"CANCEL{side.upper()}BPIC", pct(t12["cancellation_overall"], 0))]
    if comp_sep and "well_defined" in comp_sep:
        out.append(macro("WDALLSEPSIS", pct(comp_sep["well_defined"]["share_all_defined"], 0)))
    if comp_sep and "readings" in comp_sep:
        rd_sep = comp_sep["readings"]["rebuilt"]
        for side in ("act", "wait"):
            if side in rd_sep["propagation"]:
                t_sep = rd_sep["propagation"][side]
                cs_sep = t_sep["channel_share"]
                out += [macro(f"TOPFIVE{side.upper()}SEPSIS", pct(t_sep["prefix_share_top5"], 0)),
                        macro(f"CHANRISK{side.upper()}SEPSIS", pct(cs_sep["risk"])),
                        macro(f"CANCEL{side.upper()}SEPSIS", pct(t_sep["cancellation_overall"], 0))]
    bl = comp["baseline"]
    out += [macro("NBACKGROUND", str(bl["n_background"])), macro("BLRISKGAP", num(bl["risk"]["baseline_gap"], 2)), macro("BLRISKSIGN", pct(bl["risk"]["sign_agreement"], 0)),
            macro("BLRISKCORR", num(bl["risk"]["corr"], 2)), macro("BLPTGAP", num(bl["Proba_if_Treated"]["baseline_gap"], 2)), macro("BLPTSIGN", pct(bl["Proba_if_Treated"]["sign_agreement"], 0)),
            macro("BLPUGAP", num(bl["Proba_if_Untreated"]["baseline_gap"], 2)), macro("BLPUSIGN", pct(bl["Proba_if_Untreated"]["sign_agreement"], 0))]

    # --- Section 6.4: faithful? ---------------------------------------------------
    d = comp["direct"]
    ag = d["agreement_with_propagated"]
    ac = d["agreement_vs_cancellation"]
    out += [macro("AGREECANCRHO", num(ac["spearman_rho_vs_cancellation"], 2)), macro("AGREELOWCANC", num(ac["agreement_low_cancellation"], 2)),
            macro("AGREEHIGHCANC", num(ac["agreement_high_cancellation"], 2)), macro("CANCMEDIAN", pct(ac["median_cancellation"], 0)),
            macro("AGREESPEARMANACT", num(ag["act"]["spearman_mean"], 2)), macro("AGREESPEARMANWAIT", num(ag["wait"]["spearman_mean"], 2)),
            macro("AGREETOPONEACT", pct(ag["act"]["top1_agreement"], 0)), macro("AGREETOPONEWAIT", pct(ag["wait"]["top1_agreement"], 0))]
    for side in ("act", "wait"):
        gt = [a for a, _ in d["global_top"][side]]
        prop_top = rd["propagation"][side]["top"][0]["input"]
        out += [macro(f"DIRECTTOP{side.upper()}", f"\\texttt{{{tex(gt[0])}}}"),
                macro(f"DIRECTRANKOFPROPTOP{side.upper()}", f"${gt.index(prop_top) + 1}$" if prop_top in gt else f"beyond its top {len(gt)}")]
    out += [macro("NPERM", str(d["n_perm"])), macro("AGREESPEARMAN", num(ag["all"]["spearman_mean"], 2)), macro("AGREEGLOBAL", num(ag["all"]["global_spearman"], 2)),
            macro("AGREEJACCARD", num(ag["all"]["jaccard_top5_mean"], 2)), macro("AGREETOPONE", pct(ag["all"]["top1_agreement"], 0)),
            macro("DIRECTCOMPLETENESS", num(d["completeness_gap"], 1))]
    if "completeness_gap_mean" in d:
        out += [macro("DIRECTCOMPLETENESSMEAN", num(d["completeness_gap_mean"], 2)), macro("DIRECTMARGINMEAN", num(d["margin_abs_mean"], 2))]
    rows = []
    for side in ("act", "wait"):
        for k in ("1", "3"):
            r = comp["deletion_e2e"][side][k]
            rows.append(f"{side} & top-{k} & {num(r['abs_random'])} & {num(r['propagated']['abs_guided'])} & {num(r['propagated']['z'], 0)} & "
                        f"{num(r['direct']['abs_guided'])} & {num(r['direct']['z'], 0)} & {num(max(r['propagated']['abs_anti'], r['direct']['abs_anti']), 2)} \\\\ \\hline")
            for nm in ("propagated", "direct"):
                out.append(macro(f"ZE{nm.upper()}{side.upper()}K{k}", num(r[nm]["z"], 0)))
    out.append(macro("DELETIONROWS", "\n".join(rows)))
    if comp12 and "deletion_e2e" in comp12:
        zs = [comp12["deletion_e2e"][s][k]["propagated"]["z"] for s in ("act", "wait") for k in ("1", "3") if s in comp12["deletion_e2e"]]
        out.append(macro("ZEPROPMINBPIC", num(min(zs), 0)))
        ag12 = comp12["direct"]["agreement_with_propagated"]
        out += [macro("AGREESPEARMANBPIC", num(ag12["all"]["spearman_mean"], 2)),
                macro("AGREEGLOBALBPIC", num(ag12["all"]["global_spearman"], 2)),
                macro("AGREEJACCARDBPIC", num(ag12["all"]["jaccard_top5_mean"], 2)),
                macro("AGREETOPONEBPIC", pct(ag12["all"]["top1_agreement"], 0))]
    if comp_sep and "deletion_e2e" in comp_sep:
        zs_sep = [comp_sep["deletion_e2e"][s][k]["propagated"]["z"] for s in ("act", "wait") for k in ("1", "3") if s in comp_sep["deletion_e2e"]]
        if zs_sep:
            out.append(macro("ZEPROPMINSEPSIS", num(min(zs_sep), 0)))
        if "direct" in comp_sep:
            ag_sep = comp_sep["direct"]["agreement_with_propagated"]
            out += [macro("AGREESPEARMANSEPSIS", num(ag_sep["all"]["spearman_mean"], 2)),
                    macro("AGREEGLOBALSEPSIS", num(ag_sep["all"]["global_spearman"], 2)),
                    macro("AGREEJACCARDSEPSIS", num(ag_sep["all"]["jaccard_top5_mean"], 2)),
                    macro("AGREETOPONESEPSIS", pct(ag_sep["all"]["top1_agreement"], 0))]

    # --- Section 6.5: typology + disagreement -----------------------------------
    ty = rd["typology"]
    rows = []
    for rk in ("risky", "safe"):
        for tr in ("treatable", "untreatable"):
            a, w = ty[f"{rk}|{tr}|act"], ty[f"{rk}|{tr}|wait"]
            def ls(c, lv):
                return pct(c["level_share"][lv], 0) if c["n"] else "--"
            rows.append(f"{rk} & {tr} & {a['n']} & {w['n']} & {ls(a, 'effect')} / {ls(w, 'effect')} & {ls(a, 'native')} / {ls(w, 'native')} \\\\ \\hline")
    out.append(macro("TYPOLOGYROWS", "\n".join(rows)))
    for key, c in ty.items():
        stem = "TY" + re.sub(r"[^a-z]", "", key.replace("|", ""))
        out.append(macro(stem, str(c["n"])))
        if c["n"]:
            out.append(macro(stem + "EFFECT", pct(c["level_share"]["effect"], 0)))
            out.append(macro(stem + "NATIVE", pct(c["level_share"]["native"], 0)))
    n_risky = sum(ty[k]["n"] for k in ty if k.startswith("risky"))
    n_risky_act = sum(ty[k]["n"] for k in ty if k.startswith("risky") and k.endswith("act"))
    out += [macro("TYRISKYN", str(n_risky)), macro("TYRISKYACT", str(n_risky_act))]
    re_ = comp["risk_effect"]
    out += [macro("RISKEFFECTRHO", num(re_["global_spearman"], 2)), macro("RISKEFFECTJACCARD", num(re_["jaccard_top10"], 2)),
            macro("RISKEFFECTSHARED", ", \\allowbreak ".join(f"\\texttt{{{tex(s)}}}" for s in re_["shared_top"])),
            macro("RISKEFFECTSIGN", pct(re_["sign_agreement_overall"], 0) if re_["sign_agreement_overall"] is not None else "--")]
    groups = re_["groups"]
    rows = []
    for name, dd in re_["per_attribute"].items():
        if dd["sign_agreement"] is None:
            continue
        grp = next(g for g, members in groups.items() if name in members)
        rows.append(f"\\texttt{{{tex(name)}}} & ${pct(dd['risk_share'])}$ & ${pct(dd['effect_share'])}$ & ${pct(dd['sign_agreement'], 0)}$ & {grp} \\\\ \\hline")
    out.append(macro("SIGNROWS", "\n".join(rows)))
    for g in ("agree", "oppose", "independent", "one_level_only"):
        out.append(macro("SIGNN" + g.replace("_", "").upper(), str(len(groups[g]))))
        out.append(macro("SIGNLIST" + g.replace("_", "").upper(), ", \\allowbreak ".join(f"\\texttt{{{tex(a)}}}" for a in groups[g]) or "none"))
    vs_path = paths.REPO / "retrained_state_vs_shipped.json"
    if vs_path.exists():
        vs = json.loads(vs_path.read_text())[LOG]
        out += [macro("SHIPCORRR", num(vs["corr_r"], 2)), macro("SHIPCORRPT", num(vs["corr_pT"], 2)),
                macro("SHIPRULEDISAGREE", pct(1 - vs["positive_rule_agreement"], 0))]
    if comp_cate:
        cc = comp_cate[LOG]["state_agreement"]
        out += [macro("DRIFTFLIP", pct(cc["decision"]["flip_rate"], 0)), macro("DRIFTCORRR", num(cc["risk"]["corr_r_shipped"], 2)),
                macro("DRIFTCORRPT", num(cc["effect"]["corr_pT"], 2)), macro("DRIFTDEVAGREE", pct(cc["risk"]["deviation_agrees"], 0)),
                macro("DRIFTRULEAGREE", pct(cc["effect"]["positive_rule_agrees"], 0)), macro("DRIFTRISKSHARE", pct(comp_cate[LOG]["readings"]["shipped"]["level_shares"]["act"]["per_level"]["risk"], 1))]
        if "BPIC2012" in comp_cate:
            out.append(macro("DRIFTFLIPBPIC", pct(comp_cate["BPIC2012"]["state_agreement"]["decision"]["flip_rate"], 0)))
            out.append(macro("DRIFTCORRRBPIC", num(comp_cate["BPIC2012"]["state_agreement"]["risk"]["corr_r_shipped"], 1)))

    # --- Section 6.6: the cards ---------------------------------------------------
    for tag in ("act", "wait"):
        c = rd["cards"][f"paper_card_{tag}"]
        rc = risk["cards"][f"paper_card_{tag}"]
        ec = eff["cards"][f"paper_card_{tag}"]
        st = c["state"]
        tphi = c["timing_phi"]
        tot = sum(abs(v) for v in tphi.values())
        T = tag.upper()
        out += [macro(f"CARD{T}CASE", f"\\texttt{{{tex(c['case_id'])}}}"), macro(f"CARD{T}EVENT", str(c["prefix_nr"])),
                macro(f"CARD{T}RES", str(int(st["available_resources"]))), macro(f"CARD{T}PT", num(st["Proba_if_Treated"])), macro(f"CARD{T}PU", num(st["Proba_if_Untreated"])),
                macro(f"CARD{T}DQ", num(abs(c["dq"]))), macro(f"CARD{T}R", num(c["r"])), macro(f"CARD{T}CATE", f"{c['pU'] - c['pT']:+.2f}"),
                macro(f"CARD{T}EFFECTSHARE", pct((abs(tphi["Proba_if_Treated"]) + abs(tphi["Proba_if_Untreated"])) / tot, 0)),
                macro(f"CARD{T}RISKSHARE", pct((abs(tphi["reliability"]) + abs(tphi["deviation"])) / tot, 0)),
                macro(f"CARD{T}PTSHARE", pct(abs(tphi["Proba_if_Treated"]) / tot, 0)),
                macro(f"CARD{T}RESSHARE", pct(abs(tphi["available_resources"]) / tot, 0)), macro(f"CARD{T}POSSHARE", pct(abs(tphi["relative_position"]) / tot, 0)),
                macro(f"CARD{T}DEVIATION", str(int(st["deviation"]))), macro(f"CARD{T}ACTUAL", "did" if rc["actual_deviant"] else "did not"),
                macro(f"CARD{T}RISKTOP", ", ".join(f"\\texttt{{{tex(a)}}}" for a, _, _ in rc["risk_top"][:3])),
                macro(f"CARD{T}EFFECTTOP", ", ".join(f"\\texttt{{{tex(a)}}}" for a, _, _, _, _ in ec["effect_top"][:3])),
                macro(f"CARD{T}COMPOSEDTOP", ", \\allowbreak ".join(f"\\texttt{{{tex(a)}}} ({v:+.2f})" for a, v in c["composed_top"][:4])),
                macro(f"CARD{T}COMPOSEDSUM", num(c["completeness"]["composed_sum"])), macro(f"CARD{T}TIMINGSUM", num(c["completeness"]["timing_sum"]))]
        if "anchored_top" in c:
            out += [macro(f"CARD{T}ANCHOREDTOP", ", \\allowbreak ".join(f"\\texttt{{{tex(a)}}} ({v:+.2f})" for a, v in c["anchored_top"][:4])),
                    macro(f"CARD{T}DIRECTTOP", ", \\allowbreak ".join(f"\\texttt{{{tex(a)}}} ({v:+.2f})" for a, v in c["direct_top"][:4])),
                    macro(f"CARD{T}DIRECTSPEARMAN", num(c["direct_vs_propagated"]["spearman"], 2)),
                    macro(f"CARD{T}SAMETOP", "the same" if c["direct_vs_propagated"]["same_top"] else "a different")]
            fa = c["anchored_flows"][0]
            out += [macro(f"CARD{T}AFLOWTOP", f"\\texttt{{{tex(fa['input'])}}}"), macro(f"CARD{T}AFLOWTOPRISK", f"{fa['risk']:+.2f}"),
                    macro(f"CARD{T}AFLOWTOPT", f"{fa['effect_T']:+.2f}"), macro(f"CARD{T}AFLOWTOPU", f"{fa['effect_U']:+.2f}"), macro(f"CARD{T}AFLOWTOPTOTAL", f"{fa['total']:+.2f}")]
        # channel flows of the top attribute of the card
        f0 = c["flows"][0]
        out += [macro(f"CARD{T}FLOWTOP", f"\\texttt{{{tex(f0['input'])}}}"), macro(f"CARD{T}FLOWTOPRISK", f"{f0['risk']:+.2f}"),
                macro(f"CARD{T}FLOWTOPT", f"{f0['effect_T']:+.2f}"), macro(f"CARD{T}FLOWTOPU", f"{f0['effect_U']:+.2f}"), macro(f"CARD{T}FLOWTOPTOTAL", f"{f0['total']:+.2f}")]
    out += [macro("MEANCATE", num(eff["effect"]["mean_cate"])), macro("MEANR", num(risk["risk"]["mean"])), macro("EFFECTDELETIONZ", num(eff["deletion_test"]["1"]["z"], 0)),
            macro("RISKDELETIONZ", num(risk["metrics"]["deletion_test"]["1"]["z"], 0)), macro("TIMINGDELETIONZ", num(fid["wait_side"]["k1"]["gap"] / fid["wait_side"]["k1"]["gap_se"], 0))]

    # --- Sepsis provenance (own risk/effect models, own PPO agent) --------------
    sepsis_risk_manifest = paths.RISK_MODELS / "Sepsis_manifest.json"
    sepsis_eff_manifest = paths.EFFECT_MODELS / "Sepsis_manifest.json"
    if sepsis_risk_manifest.exists() and sepsis_eff_manifest.exists():
        rs = json.loads(sepsis_risk_manifest.read_text())
        es = json.loads(sepsis_eff_manifest.read_text())
        out += [macro("NTESTPREFIXESSEPSIS", f"{rs['n_test']:,}"), macro("RISKAUCSEPSIS", num(rs["test_auc"], 2)),
                macro("EFFECTAUCTSEPSIS", num(es["auc_treated_arm_on_treated_test"], 2)),
                macro("EFFECTAUCUSEPSIS", num(es["auc_untreated_arm_on_untreated_test"], 2)),
                macro("TREATEDSHARESEPSIS", pct(es["treated_share_train"], 0))]

    if comp_sep:
        out.append(macro("NPOOLSEPSIS", str(comp_sep["n"])))
        if "direct" in comp_sep and "completeness_gap_mean" in comp_sep["direct"]:
            out.append(macro("DIRECTCOMPLETENESSSEPSIS", num(comp_sep["direct"]["completeness_gap"], 1)))
    if comp12 and "direct" in comp12:
        out.append(macro("DIRECTCOMPLETENESSBPIC", num(comp12["direct"]["completeness_gap"], 1)))
    if sepsis_manifest := (paths.variant_artifact("Sepsis", pools.DEFAULT_VARIANT, "_manifest.json")):
        if sepsis_manifest.exists():
            out.append(macro("PPOSTEPSSEPSIS", f"{json.loads(sepsis_manifest.read_text())['total_timesteps']:,}"))

    # --- how each policy acts on its pool: positive-effect rule and free staff ----
    from stable_baselines3 import PPO
    for lg, suffix in (("BPIC2017", ""), ("BPIC2012", "BPIC"), ("Sepsis", "SEPSIS")):
        if not paths.variant_model(lg, pools.DEFAULT_VARIANT).exists():
            continue
        S, _rows, fn = pools.evaluation_pool(lg, pools.DEFAULT_VARIANT)
        acts = cp.margin_of(PPO.load(str(paths.variant_model(lg, pools.DEFAULT_VARIANT)), device="cpu").policy, S) > 0
        pos = cp.positive_effect_rule(S[:, fn.index("Proba_if_Treated")], S[:, fn.index("Proba_if_Untreated")])
        out += [macro(f"ACTPOS{suffix}", pct(acts[pos].mean(), 0)), macro(f"ACTNOTPOS{suffix}", pct(acts[~pos].mean(), 0)),
                macro(f"POSSHARE{suffix}", pct(pos.mean(), 0))]
        if "available_resources" in fn:
            free = S[:, fn.index("available_resources")] > 0
            out += [macro(f"ACTNOTPOSNOSTAFF{suffix}", pct(acts[~pos & ~free].mean(), 0)), macro(f"ACTNOTPOSSTAFF{suffix}", pct(acts[~pos & free].mean(), 0))]
    for key, suffix in (("bpic2012", "BPIC"), ("sepsis", "SEPSIS")):
        if key in json.loads(paths.GAIN_JSON.read_text()):
            gg = json.loads(paths.GAIN_JSON.read_text())[key]
            out += [macro(f"GAINPOLICY{suffix}", num(gg["policy_gain"], 1)), macro(f"GAINWAIT{suffix}", num(gg["always_wait_gain"], 1)),
                    macro(f"GAININTERVENE{suffix}", num(gg["always_intervene_gain"], 1)), macro(f"GAINORACLE{suffix}", num(gg["oracle_gain"], 1))]
    out.append(macro("GAININTERVENE", num(gain["always_intervene_gain"], 1)))

    audit_path = paths.REPO / "leakage_audit.json"
    if audit_path.exists():
        audit = {x["log"]: x for x in json.loads(audit_path.read_text())}
        for lg, suffix in (("BPIC2017", ""), ("BPIC2012", "BPIC"), ("Sepsis", "SEPSIS")):
            if lg in audit:
                pr = audit[lg]["propensity"]
                out += [macro(f"OVERLAP{suffix}", pct(pr["share_in_0.05_0.95"], 0)), macro(f"OVERLAPCASES{suffix}", str(pr["cases_in_overlap"])),
                        macro(f"AUDITPASS{suffix}", "passes" if audit[lg]["pass"] else "fails")]

    target = paths.PAPER_FIGURES.parent / "numbers.tex"
    target.write_text("%% Generated by paper_numbers.py from the committed result JSONs -- do not edit by hand.\n" + "\n".join(out) + "\n")
    print(f"{len(out)} macros -> {target}")


if __name__ == "__main__":
    main()
