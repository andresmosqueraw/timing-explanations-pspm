"""Paper figure: the three explanations of *one* decision point, drawn to be
read without the method section.

Top band: which decision point this is -- the case, the events seen so far,
the decision point itself with the state the policy reads there, the policy's
decision, and a note that this is a single decision point, not the pool
average of the tables. Below: three question -> answer panels (risk, effect,
timing); every bar in the timing panel is tagged with the panel its feature
comes from, so the reader sees the timing level reading the other two.
Same data as run_effect_suite's three-level card (risk_results.json,
effect_results.json, fidelity_results.json; BPIC2017, paper_card_act).

Usage: TIMING_PAPER_FIGURES=<paper>/figures python figures/make_three_level.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import paths  # noqa: E402
import pools  # noqa: E402
from plain import label  # noqa: E402

LOG, CARD = "BPIC2017", "paper_card_act"
OUT = paths.PAPER_FIGURES
OUT.mkdir(parents=True, exist_ok=True)

risk = {x["log"]: x for x in json.loads(paths.RISK_JSON.read_text())["results"]}[LOG]
eff = {x["log"]: x for x in json.loads(paths.EFFECT_JSON.read_text())["results"]}[LOG]
fid = {x["log"]: x for x in json.loads(paths.FIDELITY_JSON.read_text())}[LOG]
rc, ec = risk["cards"][CARD], eff["cards"][CARD]
acts = rc["dq"] > 0
n_side = fid["n_intervene_states"] if acts else fid["n_wait_states"]

POS, NEG, INK, MUTED, LINE = "#c2410c", "#1d4ed8", "#1f2937", "#6b7280", "#d1d5db"
LEVEL = {"risk": "#7c3aed", "effect": "#0f766e", "case": "#6b7280", "timing": "#1f2937"}
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": LINE, "xtick.color": MUTED, "ytick.color": INK})


def case_trace(case_id: str, prefix_nr: int) -> list[tuple[str, str]]:
    """(time, activity) of the events seen at the decision point, cached
    under figures/out (reading the prepared log takes a while)."""
    cache = paths.REPO / "figures/out" / f"trace_{case_id}_{prefix_nr}.json"
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text())]
    import risk_model as rm

    df, conf = rm.load_events(LOG)
    d = df[df[conf["case_col"]] == case_id].sort_values([conf["ts_col"], conf["activity_col"]], kind="mergesort").head(prefix_nr)
    trace = [(t.strftime("%H:%M:%S"), a) for t, a in zip(d[conf["ts_col"]], d[conf["activity_col"]])]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(trace))
    return trace


trace = case_trace(rc["case_id"], rc["prefix_nr"])
st = rc["state"]

fig = plt.figure(figsize=(13.0, 6.3))


def box(x, y, w, h, fc="white", ec=INK, lw=0.8, ls="-", z=1):
    fig.patches.append(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.006", transform=fig.transFigure,
                                              fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def arrow(x0, x1, y, ls="-", color=MUTED):
    fig.patches.append(patches.FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=11,
                                               color=color, lw=1.0, ls=ls, zorder=2))


# --- top band: which decision point ------------------------------------------
TOP0, TOP1 = 0.735, 0.985
box(0.012, TOP0, 0.976, TOP1 - TOP0, fc="#f8fafc", ec=LINE)
fig.text(0.024, 0.955, "Which decision point is explained?", fontsize=12.5, fontweight="bold", color=INK, va="center")
fig.text(0.024, 0.918, f"Case {rc['case_id']} (BPIC2017 test split), right after its event {rc['prefix_nr']}", fontsize=10.2, color=MUTED, va="center")

y_tl = 0.835
x = 0.024
# long prefixes: the first event, how many are skipped, and the last one
shown = trace if len(trace) <= 3 else [trace[0], ("", f"... {len(trace) - 2} more events ..."), trace[-1]]
for t, a in shown:
    w = 0.012 + 0.0068 * max(len(a), 8)
    box(x, y_tl - 0.032, w, 0.064, fc="white" if t else "#f8fafc", ec=LINE)
    fig.text(x + w / 2, y_tl + (0.012 if t else 0.0), a, fontsize=9.6, color=INK if t else MUTED, ha="center", va="center", style="normal" if t else "italic")
    if t:
        fig.text(x + w / 2, y_tl - 0.016, t, fontsize=8.4, color=MUTED, ha="center", va="center")
    arrow(x + w + 0.003, x + w + 0.02, y_tl)
    x += w + 0.023
# the decision point itself
w_dp = 0.245
box(x, y_tl - 0.058, w_dp, 0.116, fc="#fef3c7", ec="#f59e0b", lw=1.4)
fig.text(x + w_dp / 2, y_tl + 0.034, "Decision point: act now or wait?", fontsize=10.4, fontweight="bold", color=INK, ha="center", va="center")
res = int(round(st["available_resources"]))
fig.text(x + w_dp / 2, y_tl + 0.0, f"{res} of {pools.N_RESOURCES} staff free; {label('relative_position', st['relative_position'])}", fontsize=9.3, color=INK, ha="center", va="center")
fig.text(x + w_dp / 2, y_tl - 0.032, f"risk {rc['r']:.0%}; rejection {ec['pU']:.0%} without, {ec['pT']:.0%} with another offer",
         fontsize=7.4, color=MUTED, ha="center", va="center")
x_dec = x + w_dp
# the decision, and the unknown rest of the case
arrow(x_dec + 0.004, x_dec + 0.03, y_tl, color="#f59e0b")
w_d = 0.108
box(x_dec + 0.033, y_tl - 0.032, w_d, 0.064, fc=POS if acts else NEG, ec=POS if acts else NEG)
fig.text(x_dec + 0.033 + w_d / 2, y_tl + 0.009, "policy: ACT NOW" if acts else "policy: WAIT", fontsize=10.4, fontweight="bold", color="white", ha="center", va="center")
fig.text(x_dec + 0.033 + w_d / 2, y_tl - 0.016, "(make a further offer)" if acts else "(no offer yet)", fontsize=8.6, color="white", ha="center", va="center")
x_rest = x_dec + 0.033 + w_d + 0.012
arrow(x_rest, x_rest + 0.035, y_tl, ls="--")
fig.text(x_rest + 0.04, y_tl, "rest of the case:\nnot yet known", fontsize=9.2, color=MUTED, va="center", style="italic")

fig.text(0.024, 0.762,
         f"One decision point, not an average: it is 1 of the {n_side:,} decisions to {'act' if acts else 'wait'} in the BPIC2017 test pool, "
         f"whose averages Tables 2 and 4 report.", fontsize=9.6, color=INK, va="center", style="italic")

# --- three panels: question -> answer -----------------------------------------
COL_W, GAP, X0 = 0.318, 0.011, 0.012
BOX_Y0, BOX_Y1 = 0.125, 0.715
cols = [X0 + i * (COL_W + GAP) for i in range(3)]
LEFT = (0.17, 0.17, 0.2)  # room for the tick labels; panel 3's carry the value they read
axes = [fig.add_axes([c + lf, 0.215, COL_W - lf - 0.02, 0.335], zorder=2) for c, lf in zip(cols, LEFT)]
for c in cols:
    box(c, BOX_Y0, COL_W, BOX_Y1 - BOX_Y0, fc="white", ec=LINE, z=-2)

cate = ec["pU"] - ec["pT"]
positive_rule = ec["pT"] < 0.5 <= ec["pU"]
risk_answer = f"{'Yes' if rc['r'] >= 0.5 else 'Not much'}: {rc['r']:.0%} chance the loan is not accepted"
effect_answer = (f"{'Yes' if cate > 0 else 'No'}: rejection {ec['pU']:.0%} without it, {ec['pT']:.0%} with it")
timing_answer = f"{'Act now' if acts else 'Wait'}: margin for acting {rc['dq']:+.2f}"

risk_items = [(label(a, v), c, "risk") for a, v, c in rc["risk_top"]]
eff_items = [(label(a, v), c, "effect") for a, v, c, _, _ in ec["effect_top"]]


def state_label(k, v):
    return {"Proba_if_Untreated": (f"rejection without offer: {v:.0%}", "effect"),
            "Proba_if_Treated": (f"rejection with offer: {v:.0%}", "effect"),
            "reliability": (f"risk model's confidence: {v:.0%}", "risk"),
            "deviation": ("risk model: will end well" if v == 1 else "risk model: will end badly", "risk"),
            "available_resources": (f"{int(round(v))} of {pools.N_RESOURCES} staff free", "case"),
            "relative_position": (f"progress: event {rc['prefix_nr']} of a typical {int(round(rc['prefix_nr'] / v)) if v > 0 else '?'}", "case")}[k]


tk = sorted(rc["timing_phi"], key=lambda k: -abs(rc["timing_phi"][k]))
tim_items = [(*state_label(k, st[k]), ) for k in tk]
tim_items = [(lab, rc["timing_phi"][k], lvl) for (lab, lvl), k in zip(tim_items, tk)]

panels = [
    (axes[0], risk_items, "risk", "1  Is the case at risk?", risk_answer, "risk model, over the case attributes", "← lowers the risk      raises the risk →", False),
    (axes[1], eff_items, "effect", "2  Would another offer help?", effect_answer, "effect model, over the case attributes", "← helps less      helps more →", False),
    (axes[2], tim_items, "timing", f"3  Why {'act now' if acts else 'wait'}, not later?", timing_answer, "policy, over panels 1, 2 and the case", "← reason to wait      reason to act now →", True),
]
TAG = {"risk": "from 1", "effect": "from 2", "case": "case"}
for (ax, items, lvl, title, answer, who, xlabel, tagged), c in zip(panels, cols):
    phi = np.array([p for _, p, _ in items])
    ax.barh(range(len(items)), phi, color=[POS if v > 0 else NEG for v in phi], height=0.62)
    ax.set_yticks(range(len(items)), [lab for lab, _, _ in items], fontsize=8.9)
    if tagged:
        for tick, (_, _, l) in zip(ax.get_yticklabels(), items):
            tick.set_color(LEVEL[l])
    ax.invert_yaxis(); ax.axvline(0, color=INK, lw=0.8)
    ax.tick_params(axis="x", labelsize=8.6); ax.tick_params(axis="y", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # header chip, question, answer, who is explained
    box(c + 0.01, 0.668, 0.012, 0.03, fc=LEVEL[lvl], ec=LEVEL[lvl])
    fig.text(c + 0.028, 0.683, title, fontsize=12.2, fontweight="bold", color=INK, va="center")
    fig.text(c + 0.012, 0.638, answer, fontsize=10.2, color=LEVEL[lvl] if lvl != "timing" else (POS if acts else NEG), fontweight="bold", va="center")
    fig.text(c + 0.012, 0.605, f"explains the {who}", fontsize=8.8, color=MUTED, va="center")
    fig.text(c + COL_W / 2, 0.147, xlabel, fontsize=9.4, color=INK, ha="center", va="center")
# legend for the tags of panel 3
lx = cols[2] + 0.012
for i, l in enumerate(("risk", "effect", "case")):
    xx = lx + i * 0.095
    box(xx, 0.563, 0.01, 0.022, fc=LEVEL[l], ec=LEVEL[l])
    fig.text(xx + 0.014, 0.574, {"risk": "from panel 1", "effect": "from panel 2", "case": "from the case"}[l], fontsize=8.6, color=LEVEL[l], va="center")

# --- one sentence -------------------------------------------------------------
risk_s = "the loan looks likely to fail" if rc["r"] >= 0.5 else "the loan looks likely to go through"
effect_s = "another offer is expected to turn the outcome around" if positive_rule else ("another offer is expected to help" if cate > 0 else "another offer is not expected to help")
native = [(k, rc["timing_phi"][k]) for k in ("available_resources", "relative_position")]
top_native = max(native, key=lambda kv: abs(kv[1]))
reason = label(top_native[0], st[top_native[0]])
if top_native[1] > 0:
    timing_s = f"and {reason}: act now." if acts else f"but {reason}: wait."
else:
    timing_s = f"even though {reason}: act now." if acts else f"even though {reason}: wait."
sentence = f"In plain terms, at this decision point:  {risk_s}, {effect_s}, {timing_s}"
box(X0, 0.025, 3 * COL_W + 2 * GAP, 0.075, fc="#fffbeb", ec="#f59e0b", lw=0.9, z=3)
fig.text(X0 + 0.015, 0.0625, sentence, fontsize=10.8, color=INK, va="center", zorder=4)

for stem in ("fig5_three_level_card",):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=170)
print("three-level card ->", OUT / "fig5_three_level_card.pdf")
print(sentence)
