"""Paper figure: how one decision composes -- the timing justification of a
decision point propagated back to the prefix attributes through the risk and
effect explanations (compose.propagate), drawn as a flow: prefix attributes
(left) -> the six state coordinates grouped by level (middle) -> the policy's
margin (right). Ribbon width = |contribution| in units of the margin; colour =
direction (toward the decision the policy took, or against it). Data:
compose_results.json, the paper's decision to act on BPIC2017.

Usage: TIMING_PAPER_FIGURES=<paper>/figures python figures/make_composition_flow.py [--log BPIC2017] [--card paper_card_act] [--reading rebuilt]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import paths  # noqa: E402

POS, NEG, INK, MUTED, LINE = "#c2410c", "#1d4ed8", "#1f2937", "#6b7280", "#d1d5db"
LEVEL_COLOR = {"risk": "#e0e7ff", "effect": "#ffedd5", "native": "#dcfce7"}
MID = [  # (node key, label, level)
    ("risk", "reliability, deviation (risk)", "risk"),
    ("effect_T", "$\\hat p_{\\mathrm{treated}}$ (effect)", "effect"),
    ("effect_U", "$\\hat p_{\\mathrm{untreated}}$ (effect)", "effect"),
    ("relative_position", "relative_position (native)", "native"),
    ("available_resources", "available_resources (native)", "native"),
]


def fmt(v):
    if isinstance(v, str):
        return v
    return f"{v:.3g}" if abs(v - round(v)) > 1e-9 else f"{int(round(v))}"


def ribbon(ax, x0, y0, h0, x1, y1, h1, color, alpha=0.55):
    """A band from (x0, y0..y0+h0) to (x1, y1..y1+h1) with cubic ends."""
    cx = (x0 + x1) / 2
    verts = [(x0, y0), (cx, y0), (cx, y1), (x1, y1), (x1, y1 + h1), (cx, y1 + h1), (cx, y0 + h0), (x0, y0 + h0), (x0, y0)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="none", alpha=alpha, zorder=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="BPIC2017")
    ap.add_argument("--card", default="paper_card_act")
    ap.add_argument("--reading", default="rebuilt", choices=["shipped", "rebuilt"])
    ap.add_argument("--stem", default="fig6_composition_flow")
    a = ap.parse_args(argv)

    res = {x["log"]: x for x in json.loads(paths.COMPOSE_JSON.read_text())["results"]}[a.log]
    card = res["readings"][a.reading]["cards"][a.card]
    acts = card["action"] == "intervene"
    tphi = card["timing_phi"]
    flows = card["flows"]

    # --- flows attribute -> middle node, and middle node -> margin ------------
    left = [(f["input"], f.get("value"), {"risk": f["risk"], "effect_T": f["effect_T"], "effect_U": f["effect_U"]}) for f in flows]
    mid_in = {"risk": tphi["reliability"] + tphi["deviation"], "effect_T": tphi["Proba_if_Treated"], "effect_U": tphi["Proba_if_Untreated"],
              "relative_position": tphi["relative_position"], "available_resources": tphi["available_resources"]}
    mid_abs = {"risk": abs(tphi["reliability"]) + abs(tphi["deviation"]), "effect_T": abs(tphi["Proba_if_Treated"]), "effect_U": abs(tphi["Proba_if_Untreated"]),
               "relative_position": abs(tphi["relative_position"]), "available_resources": abs(tphi["available_resources"])}
    # node heights: middle = total |flow| through the node (attribute side for propagated ones)
    mid_h = {k: sum(abs(c[k]) for _, _, c in left) if k in ("risk", "effect_T", "effect_U") else mid_abs[k] for k, _, _ in MID}
    left_h = [sum(abs(v) for v in c.values()) for _, _, c in left]
    total = sum(mid_h.values())
    scale = 1.0 / total if total else 1.0

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(-0.2, 1.22); ax.axis("off")
    TOPY = 0.9  # top of the usable span; headers sit above it
    X0, X1, X2, W = 2.55, 5.55, 8.55, 0.32
    gap_l, gap_m = 0.02, 0.045

    # left column layout (top to bottom in the order of the flows list)
    hl = [h * scale for h in left_h]
    span_l = sum(hl) + gap_l * (len(hl) - 1)
    y = TOPY - (TOPY - span_l) / 2
    left_pos = []
    for h in hl:
        left_pos.append((y - h, h)); y -= h + gap_l
    # middle column layout
    hm = [mid_h[k] * scale for k, _, _ in MID]
    span_m = sum(hm) + gap_m * (len(hm) - 1)
    y = TOPY - (TOPY - span_m) / 2
    mid_pos = {}
    for (k, _, _), h in zip(MID, hm):
        mid_pos[k] = (y - h, h); y -= h + gap_m
    # right node
    right_h = sum(mid_abs.values()) * scale
    right_pos = (TOPY / 2 - right_h / 2, right_h)

    # ribbons attribute -> middle, stacked inside both nodes
    cursor_l = [p[0] + p[1] for p in left_pos]
    cursor_m = {k: p[0] + p[1] for k, p in mid_pos.items()}
    for i, (name, val, c) in enumerate(left):
        for k in ("risk", "effect_T", "effect_U"):
            h = abs(c[k]) * scale
            if h <= 0:
                continue
            color = POS if c[k] > 0 else NEG
            ribbon(ax, X0 + W, cursor_l[i] - h, h, X1, cursor_m[k] - h, h, color)
            cursor_l[i] -= h; cursor_m[k] -= h
    # ribbons middle -> margin
    cursor_r = right_pos[0] + right_pos[1]
    for k, _, _ in MID:
        h = mid_abs[k] * scale
        if h <= 0:
            continue
        color = POS if mid_in[k] > 0 else NEG
        y0 = mid_pos[k][0]  # native nodes: whole node; lower-level nodes: same height as the node
        ribbon(ax, X1 + W, y0, h, X2, cursor_r - h, h, color)
        cursor_r -= h

    # nodes and labels
    # left labels: centred on the node, spread to at least MINSEP_L apart (top-down), leader line when moved
    MINSEP_L = 0.06
    lys = []
    for y0, h in left_pos:
        c = y0 + h / 2
        lys.append(c if not lys else min(c, lys[-1] - MINSEP_L))
    if lys and lys[-1] < -0.02:
        shift = -0.02 - lys[-1]
        lys = [y + shift for y in lys]
    for (name, val, c), (y0, h), ly in zip(left, left_pos, lys):
        ax.add_patch(plt.Rectangle((X0, y0), W, h, fc="#f3f4f6", ec=INK, lw=0.6, zorder=2))
        tot = sum(c.values())
        lab = name if val is None else f"{name} = {fmt(val)}"
        if abs(ly - (y0 + h / 2)) > 1e-6:
            ax.plot([X0 - 0.06, X0], [ly, y0 + h / 2], color=MUTED, lw=0.5, zorder=2)
        ax.text(X0 - 0.08, ly, lab, ha="right", va="center", fontsize=7.0, color=INK)
        ax.text(X0 + W + 0.06, ly, f"{tot:+.2f}", ha="left", va="center", fontsize=6.4, color=MUTED)
    # middle labels: centred on their node, then spread so that consecutive
    # labels are at least MINSEP apart and the stack stays inside [0, TOPY]
    MINSEP = 0.13
    centers = [mid_pos[k][0] + mid_pos[k][1] / 2 for k, _, _ in MID]
    ys = [min(centers[0], TOPY - 0.02)]
    for c in centers[1:]:
        ys.append(min(c, ys[-1] - MINSEP))
    if ys[-1] < 0.03:
        shift = 0.03 - ys[-1]
        ys = [y + shift for y in ys]
    for (k, lab, lvl), ly in zip(MID, ys):
        y0, h = mid_pos[k]
        ax.add_patch(plt.Rectangle((X1, y0), W, h, fc=LEVEL_COLOR[lvl], ec=INK, lw=0.6, zorder=2))
        ax.plot([X1 + W, X1 + W + 0.06], [y0 + h / 2, ly], color=MUTED, lw=0.5, zorder=2)
        ax.text(X1 + W + 0.08, ly, f"{lab}\n$\\phi^{{\\Delta Q}}$ = {mid_in[k]:+.2f}", ha="left", va="center", fontsize=6.9, color=INK)
    y0, h = right_pos
    ax.add_patch(plt.Rectangle((X2, y0), W, h, fc="#fef3c7", ec=INK, lw=0.8, zorder=2))
    mlabel = f"$\\Delta Q$ = {card['dq']:.2f}\nact now" if acts else f"$\\Delta Q_{{\\mathrm{{wait}}}}$ = {-card['dq']:.2f}\nwait"
    ax.text(X2 + W + 0.08, TOPY / 2, mlabel, ha="left", va="center", fontsize=8, color=INK, fontweight="bold")

    # headers
    ax.text(X0 + W / 2, 1.13, "prefix attributes\n(risk expl. $\\phi^{r}$, effect expl. $\\phi^{p_T}$, $\\phi^{p_U}$)", ha="center", va="center", fontsize=7.6, color=MUTED)
    ax.text(X1 + W / 2, 1.13, "state coordinates, by level", ha="center", va="center", fontsize=7.6, color=MUTED)
    ax.text(X2 + W / 2, 1.13, "timing level", ha="center", va="center", fontsize=7.6, color=MUTED)
    side = "toward acting now" if acts else "toward waiting"
    ax.text(0.05, -0.16, f"ribbon width = |contribution| in units of the margin;  orange = {side},  blue = against it.  "
            f"Case {card['case_id']}, event {card['prefix_nr']}; $r$ = {card['r']:.2f}, $\\hat p_T$ = {card['pT']:.2f}, $\\hat p_U$ = {card['pU']:.2f}.",
            ha="left", va="center", fontsize=6.8, color=MUTED)

    out = paths.PAPER_FIGURES
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{a.stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{a.stem}.png", dpi=170, bbox_inches="tight")
    print("composition flow ->", out / f"{a.stem}.pdf")


if __name__ == "__main__":
    main()
