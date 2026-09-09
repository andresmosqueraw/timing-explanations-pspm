"""Paper figure: the multi-level explanation as a monitoring application would
show it to a process owner, in its simplest form -- one case, one decision,
three cards, one sentence, and the auditor's verdict. Every number is a real
result on BPIC2017 (risk_results.json, effect_results.json when present,
fidelity_results.json).

Usage: TIMING_PAPER_FIGURES=<paper>/figures python figures/make_mockup.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

OUT = paths.PAPER_FIGURES
OUT.mkdir(parents=True, exist_ok=True)

risk = {x["log"]: x for x in json.loads(paths.RISK_JSON.read_text())["results"]}["BPIC2017"]
card = risk["cards"]["paper_card_act"]
effect, effect_mean = None, None
if paths.EFFECT_JSON.exists():
    e = {x["log"]: x for x in json.loads(paths.EFFECT_JSON.read_text())["results"]}
    if "BPIC2017" in e and "paper_card_act" in e["BPIC2017"]["cards"]:
        effect = e["BPIC2017"]["cards"]["paper_card_act"]
        effect_mean = e["BPIC2017"]["effect"]["mean_cate"]
fid = {x["log"]: x for x in json.loads(paths.FIDELITY_JSON.read_text())}["BPIC2017"]
k1 = fid["intervene_side"]["k1"]
st = card["state"]

from plain import label as pretty  # noqa: E402  (plain-language readings shared with make_three_level.py)

POS, NEG, INK, MUTED, SURF, PANEL, LINE, ACCENT, WARN = "#c2410c", "#1d4ed8", "#1f2937", "#6b7280", "#f5f6f8", "#ffffff", "#d1d5db", "#0f766e", "#b45309"
plt.rcParams.update({"font.family": "DejaVu Sans"})

fig = plt.figure(figsize=(9.0, 4.7))
fig.patch.set_facecolor(SURF)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")


def box(x, y, w, h, fc=PANEL, ec=LINE, lw=0.8, r=0.7, z=1):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def text(x, y, s, size=8, color=INK, weight="normal", ha="left", va="center", z=5, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z, **kw)


def pill(x, y, w, h, label, color, size=6.5):
    box(x, y, w, h, fc=color, ec=color, r=0.5, z=3)
    text(x + w / 2, y + h / 2, label, size=size, color="white", ha="center", weight="bold", z=6)


def bars(x0, y0, w, h, items):
    """Up to three reasons, one bar each: a plain label above, a bar below, no numbers."""
    n = len(items)
    row = h / n
    vmax = max(abs(p) for _, p in items) or 1.0
    for i, (lab, p) in enumerate(items):
        top = y0 + h - row * i
        text(x0, top - row * 0.28, lab, size=7.4, color=INK)
        bw = abs(p) / vmax * (w - 1)
        ax.add_patch(patches.FancyBboxPatch((x0, top - row * 0.86), bw, row * 0.38, boxstyle="round,pad=0,rounding_size=0.15",
                                            fc=POS if p >= 0 else NEG, ec="none", zorder=4))


# --- header: the case and the decision ---------------------------------------
box(1.5, 42, 97, 8.7)
text(3, 47.9, "Loan application 788777485  ·  event 21", size=9.5, weight="bold")
text(3, 44.4, "Process owner view", size=7, color=MUTED)
pill(60, 43.6, 16, 5.4, "ACT NOW", POS, size=10)
text(77.5, 47.3, "call the applicant today", size=8.2)
text(77.5, 44.6, f"margin for acting  {card['dq']:.1f}", size=7, color=MUTED)

# --- three cards --------------------------------------------------------------
cw, ch, cy = 31.3, 27.0, 12.5
timing_keys = sorted(card["timing_phi"], key=lambda k: -abs(card["timing_phi"][k]))[:3]
cards = [
    ("Why is this case at risk?", f"{card['r']:.0%} chance the loan is not accepted",
     [(pretty(a, v), c) for a, v, c in card["risk_top"][:3]]),
    ("Why would calling help?",
     (f"estimated effect of the call {effect['pT'] - effect['pU']:.2f}, usual {effect_mean:.2f}" if effect else "estimated effect of the call"),
     ([(pretty(a, v), c) for a, v, c, _, _ in effect["effect_top"][:3]] if effect else None)),
    ("Why call now, not later?", "what tipped the policy to act",
     [(pretty(k, card["state"][k]), card["timing_phi"][k]) for k in timing_keys]),
]
for i, (title, sub, items) in enumerate(cards):
    x0 = 1.5 + i * (cw + 1.55)
    box(x0, cy, cw, ch)
    pill(x0 + 1.2, cy + ch - 3.7, 2.7, 2.7, str(i + 1), INK, size=7.5)
    text(x0 + 4.8, cy + ch - 2.35, title, size=8.8, weight="bold")
    text(x0 + 1.2, cy + ch - 5.8, sub, size=7, color=MUTED)
    if items:
        bars(x0 + 1.4, cy + 2.2, cw - 2.8, ch - 9.2, items)
    else:
        text(x0 + cw / 2, cy + ch / 2 - 1.5, "(effect card: estimator\nstill training)", size=7, color=MUTED, ha="center")

# --- one sentence -------------------------------------------------------------
box(1.5, 6.5, 97, 4.8, fc="#fffbeb", ec="#f59e0b", lw=0.8)
text(3, 8.9, "In plain terms:  the loan looks likely to fail, a call is expected to change that, and someone is free to make it.  Act now.", size=7.9)

# --- auditor and analyst strips -----------------------------------------------
box(1.5, 1.0, 47.5, 4.6)
text(3, 4.6, "Auditor view", size=6.5, color=MUTED)
pill(3, 1.7, 2.6, 1.7, "✓", ACCENT, size=6)
text(6.3, 2.55, f"faithful: removing the top reason reverses {k1['flip_guided']:.0%} of the calls  ·  4 methods agree", size=6.6)

box(51, 1.0, 47.5, 4.6)
text(52.5, 4.6, "Analyst view  ·  this period", size=6.5, color=MUTED)
sh = dict(zip(fid["feature_names"], fid["intervene_side"]["phi_share"]))
top = sorted(sh, key=lambda k: -sh[k])[:2]
short = {"available_resources": "free staff", "Proba_if_Treated": "outcome if called", "Proba_if_Untreated": "outcome if not called",
         "relative_position": "case progress", "reliability": "confidence", "deviation": "predicted deviation"}
text(52.5, 2.55, f"calls rest on {short[top[0]]} ({sh[top[0]]:.0%}) and {short[top[1]]} ({sh[top[1]]:.0%}) across 500 decisions", size=6.6)

fig.savefig(OUT / "fig6_web_mockup.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig6_web_mockup.png", dpi=170, bbox_inches="tight")
print("mockup done ->", OUT / "fig6_web_mockup.pdf", "| effect card:", "real" if effect else "placeholder")
