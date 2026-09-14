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
    ("effect_T", "Treated-arm probability (effect)", "effect"),
    ("effect_U", "Untreated-arm probability (effect)", "effect"),
    ("relative_position", "relative_position (native)", "native"),
    ("available_resources", "available_resources (native)", "native"),
]


def _collapse_to_top(flows, keep):
    """Keep the `keep` largest-magnitude named inputs and fold everything else -- including
    the JSON's own "(other attributes)" bucket -- into one aggregated row, so the figure
    names only what matters and does not try to label every input individually."""
    def mag(f):
        return abs(f["risk"]) + abs(f["effect_T"]) + abs(f["effect_U"])

    named = [f for f in flows if f["input"] != "(other attributes)"]
    top = sorted(named, key=mag, reverse=True)[:keep]
    top_names = {f["input"] for f in top}
    rest = [f for f in flows if f["input"] not in top_names]
    agg = {"input": "(other attributes)", "value": None,
           "risk": sum(f["risk"] for f in rest), "effect_T": sum(f["effect_T"] for f in rest),
           "effect_U": sum(f["effect_U"] for f in rest)}
    return sorted(top, key=mag, reverse=True) + [agg]


def fmt(v):
    if isinstance(v, str):
        return v
    return f"{v:.3g}" if abs(v - round(v)) > 1e-9 else f"{int(round(v))}"


def ribbon(ax, x0, y0, h0, x1, y1, h1, color, alpha=0.32):
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
    ap.add_argument("--source", default="propagated", choices=["propagated", "anchored"],
                    help="propagated = Eq. 1 (aggregate operator); anchored = the per-decision variant (direct ranking of F, propagated channel split)")
    ap.add_argument("--stem", default="fig6_composition_flow")
    a = ap.parse_args(argv)

    res = {x["log"]: x for x in json.loads(paths.COMPOSE_JSON.read_text())["results"]}[a.log]
    card = res["readings"][a.reading]["cards"][a.card]
    acts = card["action"] == "intervene"
    tphi = card["timing_phi"]
    flows = card["anchored_flows"] if a.source == "anchored" else card["flows"]
    flows = _collapse_to_top(flows, keep=4)  # keep the figure legible: name only the largest inputs, group the rest

    # --- flows attribute -> middle node, and middle node -> margin ------------
    left = [(f["input"], f.get("value"), {"risk": f["risk"], "effect_T": f["effect_T"], "effect_U": f["effect_U"]}) for f in flows]
    mid_in = {"risk": tphi["reliability"] + tphi["deviation"], "effect_T": tphi["Proba_if_Treated"], "effect_U": tphi["Proba_if_Untreated"],
              "relative_position": tphi["relative_position"], "available_resources": tphi["available_resources"]}
    mid_abs = {"risk": abs(tphi["reliability"]) + abs(tphi["deviation"]), "effect_T": abs(tphi["Proba_if_Treated"]), "effect_U": abs(tphi["Proba_if_Untreated"]),
               "relative_position": abs(tphi["relative_position"]), "available_resources": abs(tphi["available_resources"])}
    # node heights: middle = total |flow| through the node (attribute side for propagated ones)
    mid_h = {k: sum(abs(c[k]) for _, _, c in left) if k in ("risk", "effect_T", "effect_U") else mid_abs[k] for k, _, _ in MID}
    if a.source == "anchored":
        # the anchored card lives on the direct attribution of F: the channel sums
        # over *all* attributes (the flows include the aggregated rest) replace the
        # timing coordinates' phi, and the natives come from the direct attribution
        for k in ("risk", "effect_T", "effect_U"):
            tot = sum(f[k] for f in flows)
            mid_in[k], mid_abs[k] = tot, abs(tot)
        direct = dict(card["direct_top"])
        for k in ("relative_position", "available_resources"):
            v = direct.get(k, 0.0)
            mid_in[k], mid_abs[k], mid_h[k] = v, abs(v), abs(v)
    left_h = [sum(abs(v) for v in c.values()) for _, _, c in left]
    total = sum(mid_h.values())
    scale = 1.0 / total if total else 1.0

    fig, ax = plt.subplots(figsize=(13.0, 4.0))
    ax.set_xlim(0, 10); ax.set_ylim(-0.28, 1.26); ax.axis("off")
    TOPY = 0.9  # top of the usable span; headers sit above it
    X0, X1, X2, W = 1.3, 5.15, 9.0, 0.34
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
    MINSEP_L = 0.095
    lys = []
    for y0, h in left_pos:
        c = y0 + h / 2
        lys.append(c if not lys else min(c, lys[-1] - MINSEP_L))
    if lys and lys[-1] < -0.05:  # bottom overflow: push the whole stack up (leader lines absorb the offset)
        shift = min(-0.05 - lys[-1], 1.05 - lys[0])
        lys = [y + shift for y in lys]
    for (name, val, c), (y0, h), ly in zip(left, left_pos, lys):
        ax.add_patch(plt.Rectangle((X0, y0), W, h, fc="#f3f4f6", ec=INK, lw=0.6, zorder=2))
        # categorical aggregates are the running lexicographic maximum (risk_model.encode_prefixes), not the last value
        lab = name if val is None else (f"{name} (max so far) = {fmt(val)}" if isinstance(val, str) and name != "(other attributes)" else f"{name} = {fmt(val)}")
        if abs(ly - (y0 + h / 2)) > 1e-6:
            ax.plot([X0 - 0.06, X0], [ly, y0 + h / 2], color=MUTED, lw=0.5, zorder=2)
        ax.text(X0 - 0.08, ly, lab, ha="right", va="center", fontsize=10.2, color=INK)
    # middle labels: centred on their node, then spread so that consecutive
    # labels are at least MINSEP apart and the stack stays inside [0, TOPY]
    MINSEP = 0.1
    centers = [mid_pos[k][0] + mid_pos[k][1] / 2 for k, _, _ in MID]
    ys = [min(centers[0], TOPY - 0.06)]
    for c in centers[1:]:
        ys.append(min(c, ys[-1] - MINSEP))
    if ys[-1] < 0.03:
        shift = 0.03 - ys[-1]
        ys = [y + shift for y in ys]
    for (k, lab, lvl), ly in zip(MID, ys):
        y0, h = mid_pos[k]
        ax.add_patch(plt.Rectangle((X1, y0), W, h, fc=LEVEL_COLOR[lvl], ec=INK, lw=0.6, zorder=2))
        ax.plot([X1 + W, X1 + W + 0.06], [y0 + h / 2, ly], color=MUTED, lw=0.5, zorder=2)
        ax.text(X1 + W + 0.08, ly, lab, ha="left", va="center", fontsize=10.1, color=INK)
    y0, h = right_pos
    ax.add_patch(plt.Rectangle((X2, y0), W, h, fc="#fef3c7", ec=INK, lw=0.8, zorder=2))
    mlabel = f"margin = {card['dq']:.2f}\nact now" if acts else f"wait margin = {-card['dq']:.2f}\nwait"
    ax.text(X2 + W + 0.08, TOPY / 2, mlabel, ha="left", va="center", fontsize=11.8, color=INK, fontweight="bold")

    # headers
    ax.text(X0 + W / 2, 1.17, "prefix attributes", ha="center", va="center", fontsize=11.1, color=MUTED)
    ax.text(X1 + W / 2, 1.17, "state coordinates, by level", ha="center", va="center", fontsize=11.1, color=MUTED)
    ax.text(X2 + W / 2, 1.17, "timing level", ha="center", va="center", fontsize=11.1, color=MUTED)
    side = "toward acting now" if acts else "toward waiting"
    ax.text((X0 + X2 + W) / 2, -0.23, f"ribbon width = |contribution|;  orange = {side},  blue = against it.",
            ha="center", va="center", fontsize=9.9, color=MUTED)

    out = paths.PAPER_FIGURES
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{a.stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{a.stem}.png", dpi=170, bbox_inches="tight")
    print("composition flow ->", out / f"{a.stem}.pdf")


if __name__ == "__main__":
    main()
