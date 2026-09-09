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
         "available_resources": r"\texttt{available\_resources}", "Proba_if_Treated": r"$\hat{p}_{\text{treated}}$",
         "Proba_if_Untreated": r"$\hat{p}_{\text{untreated}}$"}
LEVEL_NAME = {"native": "Native", "risk": "Risk", "effect": "Effect"}
CHAN_NAME = {"risk": "risk", "effect_T": r"effect ($\hat{p}_T$)", "effect_U": r"effect ($\hat{p}_U$)", "native": "native"}


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
    fid = {x["log"]: x for x in json.loads(paths.FIDELITY_JSON.read_text())}[LOG]
    comp = {x["log"]: x for x in json.loads(paths.COMPOSE_JSON.read_text())["results"]}[LOG]
    comp12 = {x["log"]: x for x in json.loads(paths.COMPOSE_JSON.read_text())["results"]}.get("BPIC2012")
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

    # --- Section 6.0 pool ------------------------------------------------------
    n_act, n_wait = fid["n_intervene_states"], fid["n_wait_states"]
    ref = dict(zip(fid["feature_names"], fid["reference"]))
    out += [macro("NACT", str(n_act)), macro("NWAIT", str(n_wait)), macro("NPOOL", str(fid["n_pool_states"])),
            macro("REFWAITMARGIN", num(-float(cp.margin_of(__import__("stable_baselines3").PPO.load(str(paths.variant_model(LOG, pools.DEFAULT_VARIANT)), device="cpu").policy,
                                                       np.array([fid["reference"]], np.float32))[0]), 2))]

    # --- Table card: level shares ------------------------------------------------
    sh = {side: fid[key]["phi_share"] for side, key in (("act", "intervene_side"), ("wait", "wait_side")) if key in fid}
    feats = fid["feature_names"]
    rows = []
    for level, fs in (("effect", ["Proba_if_Treated", "Proba_if_Untreated"]), ("risk", ["reliability", "deviation"]), ("native", ["available_resources", "relative_position"])):
        tot = {s: 0.0 for s in sh}
        for f in fs:
            i = feats.index(f)
            rows.append(f"{NAMES[f]} & {LEVEL_NAME[level]} & " + " & ".join(f"${pct(sh[s][i])}$" for s in ("act", "wait") if s in sh) + r" \\ \hline")
            for s in sh:
                tot[s] += sh[s][i]
        rows.append(r"\rowcolor{keyrow} \textbf{" + LEVEL_NAME[level] + " level total} & & " + " & ".join(f"$\\mathbf{{{pct(tot[s])}}}$" for s in ("act", "wait") if s in sh) + r" \\ \hline")
        for s in sh:
            out.append(macro(f"SHARE{level.upper()}{s.upper()}", pct(tot[s])))
    out.append(macro("CARDROWS", "\n".join(rows)))
    for f in feats:
        i = feats.index(f)
        for s in sh:
            out.append(macro(f"SHARE{f.replace('_', '').replace('Probaif', 'P').upper()}{s.upper()}", pct(sh[s][i])))

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
            rows.append(f"\\texttt{{{tex(d['input'])}}} & ${pct(d['share'])}$ & {split} \\\\ \\hline")
        out.append(macro(f"INHERIT{side.upper()}ROWS", "\n".join(rows)))
        cs = t["channel_share"]
        out += [macro(f"CHANRISK{side.upper()}", pct(cs["risk"])), macro(f"CHANEFFECTT{side.upper()}", pct(cs["effect_T"])),
                macro(f"CHANEFFECTU{side.upper()}", pct(cs["effect_U"])), macro(f"CHANEFFECT{side.upper()}", pct(cs["effect_T"] + cs["effect_U"])),
                macro(f"CHANNATIVE{side.upper()}", pct(cs["native"])), macro(f"TOPFIVE{side.upper()}", pct(t["prefix_share_top5"], 0)),
                macro(f"TOPINPUT{side.upper()}", f"\\texttt{{{tex(t['top'][0]['input'])}}}"), macro(f"TOPINPUTSHARE{side.upper()}", pct(t["top"][0]["share"], 0))]
    out += [macro("COMPLETENESSGAP", f"{rd['propagation']['completeness_gap']:.0e}".replace("e-0", "e-")),
            macro("FALLBACKS", str(sum(rd["propagation"]["fallbacks"].values()))), macro("NPREFIXATTRS", str(len(inputs) - 2))]

    # --- Section 6.4: faithful? ---------------------------------------------------
    d = comp["direct"]
    ag = d["agreement_with_propagated"]
    out += [macro("NPERM", str(d["n_perm"])), macro("AGREESPEARMAN", num(ag["all"]["spearman_mean"], 2)), macro("AGREEGLOBAL", num(ag["all"]["global_spearman"], 2)),
            macro("AGREEJACCARD", num(ag["all"]["jaccard_top5_mean"], 2)), macro("AGREETOPONE", pct(ag["all"]["top1_agreement"], 0)),
            macro("DIRECTCOMPLETENESS", f"{d['completeness_gap']:.0e}")]
    rows = []
    for side in ("act", "wait"):
        for k in ("1", "3"):
            r = comp["deletion_e2e"][side][k]
            rows.append(f"{side} & {k} & {num(r['abs_random'])} & {num(r['propagated']['abs_guided'])} & {num(r['propagated']['z'], 0)} & "
                        f"{num(r['direct']['abs_guided'])} & {num(r['direct']['z'], 0)} & {num(max(r['propagated']['abs_anti'], r['direct']['abs_anti']), 2)} \\\\ \\hline")
            for nm in ("propagated", "direct"):
                out.append(macro(f"ZE{nm.upper()}{side.upper()}K{k}", num(r[nm]["z"], 0)))
    out.append(macro("DELETIONROWS", "\n".join(rows)))
    if comp12 and "deletion_e2e" in comp12:
        zs = [comp12["deletion_e2e"][s][k]["propagated"]["z"] for s in ("act", "wait") for k in ("1", "3") if s in comp12["deletion_e2e"]]
        out.append(macro("ZEPROPMINBPIC", num(min(zs), 0)))
        out.append(macro("AGREESPEARMANBPIC", num(comp12["direct"]["agreement_with_propagated"]["all"]["spearman_mean"], 2)))

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
            macro("RISKEFFECTSHARED", ", ".join(f"\\texttt{{{tex(s)}}}" for s in re_["shared_top"])),
            macro("RISKEFFECTSIGN", pct(re_["sign_agreement_overall"], 0) if re_["sign_agreement_overall"] is not None else "--")]
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
                macro(f"CARD{T}LEN", str(int(round(c["prefix_nr"] / st["relative_position"]))) if st["relative_position"] > 0 else "?"),
                macro(f"CARD{T}RES", str(int(st["available_resources"]))), macro(f"CARD{T}PT", num(st["Proba_if_Treated"])), macro(f"CARD{T}PU", num(st["Proba_if_Untreated"])),
                macro(f"CARD{T}DQ", num(abs(c["dq"]))), macro(f"CARD{T}R", num(c["r"])), macro(f"CARD{T}CATE", f"{c['pT'] - c['pU']:+.2f}"),
                macro(f"CARD{T}EFFECTSHARE", pct((abs(tphi["Proba_if_Treated"]) + abs(tphi["Proba_if_Untreated"])) / tot, 0)),
                macro(f"CARD{T}RISKSHARE", pct((abs(tphi["reliability"]) + abs(tphi["deviation"])) / tot, 0)),
                macro(f"CARD{T}PTSHARE", pct(abs(tphi["Proba_if_Treated"]) / tot, 0)),
                macro(f"CARD{T}RESSHARE", pct(abs(tphi["available_resources"]) / tot, 0)), macro(f"CARD{T}POSSHARE", pct(abs(tphi["relative_position"]) / tot, 0)),
                macro(f"CARD{T}DEVIATION", str(int(st["deviation"]))), macro(f"CARD{T}ACTUAL", "did" if rc["actual_deviant"] else "did not"),
                macro(f"CARD{T}RISKTOP", ", ".join(f"\\texttt{{{tex(a)}}}" for a, _, _ in rc["risk_top"][:3])),
                macro(f"CARD{T}EFFECTTOP", ", ".join(f"\\texttt{{{tex(a)}}}" for a, _, _, _, _ in ec["effect_top"][:3])),
                macro(f"CARD{T}COMPOSEDTOP", ", ".join(f"\\texttt{{{tex(a)}}} ({v:+.2f})" for a, v in c["composed_top"][:4])),
                macro(f"CARD{T}COMPOSEDSUM", num(c["completeness"]["composed_sum"])), macro(f"CARD{T}TIMINGSUM", num(c["completeness"]["timing_sum"]))]
        # channel flows of the top attribute of the card
        f0 = c["flows"][0]
        out += [macro(f"CARD{T}FLOWTOP", f"\\texttt{{{tex(f0['input'])}}}"), macro(f"CARD{T}FLOWTOPRISK", f"{f0['risk']:+.2f}"),
                macro(f"CARD{T}FLOWTOPT", f"{f0['effect_T']:+.2f}"), macro(f"CARD{T}FLOWTOPU", f"{f0['effect_U']:+.2f}"), macro(f"CARD{T}FLOWTOPTOTAL", f"{f0['total']:+.2f}")]
    out += [macro("MEANCATE", num(eff["effect"]["mean_cate"])), macro("MEANR", num(risk["risk"]["mean"])), macro("EFFECTDELETIONZ", num(eff["deletion_test"]["1"]["z"], 0)),
            macro("RISKDELETIONZ", num(risk["metrics"]["deletion_test"]["1"]["z"], 0)), macro("TIMINGDELETIONZ", num(fid["wait_side"]["k1"]["gap"] / fid["wait_side"]["k1"]["gap_se"], 0))]

    target = paths.PAPER_FIGURES.parent / "numbers.tex"
    target.write_text("%% Generated by paper_numbers.py from the committed result JSONs -- do not edit by hand.\n" + "\n".join(out) + "\n")
    print(f"{len(out)} macros -> {target}")


if __name__ == "__main__":
    main()
