"""Paper figure: the multi-level explanation of one decision to act, drawn as
the process owner would read it -- the decision, the three questions, the
attributions with their technical name *and* a plain reading, and an
"in plain terms" sentence. Same data as run_effect_suite's three-level card
(risk_results.json + effect_results.json, BPIC2017, paper_card_act).

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
from plain import label, noun  # noqa: E402

LOG, CARD = "BPIC2017", "paper_card_act"
OUT = paths.PAPER_FIGURES
OUT.mkdir(parents=True, exist_ok=True)

risk = {x["log"]: x for x in json.loads(paths.RISK_JSON.read_text())["results"]}[LOG]
eff = {x["log"]: x for x in json.loads(paths.EFFECT_JSON.read_text())["results"]}[LOG]
rc, ec = risk["cards"][CARD], eff["cards"][CARD]
mean_cate = eff["effect"]["mean_cate"]
cate = ec["pT"] - ec["pU"]

POS, NEG, INK, MUTED, LINE = "#c2410c", "#1d4ed8", "#1f2937", "#6b7280", "#d1d5db"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": LINE, "xtick.color": MUTED, "ytick.color": INK})


def fmt(v):
    return v if isinstance(v, str) else (f"{v:.3g}" if abs(v - round(v)) > 1e-9 else f"{int(round(v))}")


risk_items = [(a, v, c) for a, v, c in rc["risk_top"]]
eff_items = [(a, v, c) for a, v, c, _, _ in ec["effect_top"]]
tk = sorted(rc["timing_phi"], key=lambda k: -abs(rc["timing_phi"][k]))
tim_items = [(k, rc["state"][k], rc["timing_phi"][k]) for k in tk]

fig = plt.figure(figsize=(8.8, 4.2))
# three framed panels: [box] title / subtitle / bar chart (tick labels inside the frame)
COL_W, GAP, X0 = 0.318, 0.013, 0.012
BOX_Y0, BOX_Y1 = 0.165, 0.985
cols = [X0 + i * (COL_W + GAP) for i in range(3)]
axes = [fig.add_axes([c + 0.165, 0.285, COL_W - 0.185, 0.54], zorder=2) for c in cols]
for c in cols:
    fig.patches.append(patches.FancyBboxPatch((c, BOX_Y0), COL_W, BOX_Y1 - BOX_Y0, boxstyle="square,pad=0",
                                              transform=fig.transFigure, fc="white", ec="black", lw=0.8, zorder=-2))

panels = [
    (axes[0], risk_items, "1  Why is this case at risk?", f"{rc['r']:.0%} chance the loan is not accepted", r"$\phi^{r}$: toward a bad outcome $\rightarrow$", False),
    (axes[1], eff_items, "2  Why would calling help?", f"estimated effect of the call {cate:.2f}, usual {mean_cate:.2f}", r"$\phi^{\mathrm{CATE}}$: toward a larger effect $\rightarrow$", False),
    (axes[2], tim_items, "3  Why call now, not later?", f"margin for acting $\\Delta Q$ = {rc['dq']:.2f}", r"$\phi^{\Delta Q}$: toward acting now $\rightarrow$", True),
]
for ax, items, title, sub, xlabel, is_state in panels:
    names = [a for a, _, _ in items]
    phi = np.array([c for _, _, c in items])
    ax.barh(range(len(items)), phi, color=[POS if v > 0 else NEG for v in phi], height=0.62)
    labs = [label(a, v) for a, v, _ in items]
    ax.set_yticks(range(len(items)), labs, fontsize=7.1)
    ax.set_facecolor("white")
    ax.invert_yaxis(); ax.axvline(0, color=INK, lw=0.8)
    ax.tick_params(axis="x", labelsize=7); ax.tick_params(axis="y", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    c = cols[axes.index(ax)]
    fig.text(c + COL_W / 2, 0.195, xlabel, fontsize=7.6, color=INK, ha="center", va="center")  # centred in the panel
    fig.text(c + 0.014, 0.935, title, fontsize=9.6, fontweight="bold", color=INK, ha="left", va="center")
    fig.text(c + 0.014, 0.885, sub, fontsize=7.5, color=MUTED, ha="left", va="center")
# grey the technical line of each label: draw plain reading in ink, technical in muted via two-line tick labels is
# not separately colourable, so keep both in ink but different size through the newline (matplotlib limitation).

# --- one sentence -------------------------------------------------------------
acts = rc["dq"] > 0
risk_s = "the loan looks likely to fail" if rc["r"] >= 0.5 else "the loan looks likely to go through"
effect_s = "a call is expected to help" if cate > mean_cate else "a call is not expected to help much"
native = [(k, rc["timing_phi"][k]) for k in ("available_resources", "relative_position")]
top_native = max(native, key=lambda kv: abs(kv[1]))
reason = label(top_native[0], rc["state"][top_native[0]])
timing_s = (f"and {reason}.  Act now." if acts else f"but {reason}.  Wait.")
sentence = f"In plain terms:  {risk_s}, {effect_s}, {timing_s}"
fig.patches.append(patches.FancyBboxPatch((X0, 0.03), 3 * COL_W + 2 * GAP, 0.085, boxstyle="square,pad=0", transform=fig.transFigure,
                                          fc="#fffbeb", ec="#f59e0b", lw=0.9, zorder=3))
fig.text(X0 + 0.015, 0.0725, sentence, fontsize=8.6, color=INK, va="center", zorder=4)

for stem in ("fig5_three_level_card",):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=170)
print("three-level card ->", OUT / "fig5_three_level_card.pdf")
print(sentence)
