"""The paper's four figures: the MDP timeline (Fig. 1), the BPIC2017
checkpoint's training curve (Fig. 2) and two explanation cards (Figs. 3-4)
computed on the Section 6 evaluation pool (pools.bpic_pool).

Writes to paths.PAPER_FIGURES (default figures/out; set TIMING_PAPER_FIGURES
to the manuscript's figures/ directory to regenerate them in place).

Usage: python figures/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402
import pools  # noqa: E402
from dual_level import MarginHead, WaitMarginHead, integrated_gradients  # noqa: E402

OUT = paths.PAPER_FIGURES
OUT.mkdir(parents=True, exist_ok=True)
FEATS = pools.FEATS

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

# ---------------------------------------------------------------------
# Fig 1: conceptual MDP timeline (wait-vs-act repeated decision)
# ---------------------------------------------------------------------
WAIT_COLOR = "#c44e52"
INTERVENE_COLOR = "#2e8b57"
NODE_EDGE = "#1a252c"
TEXT_COLOR = "#1a252c"

fig, ax = plt.subplots(figsize=(6.0, 2.85))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

xs = [0, 1.5, 3, 4.5, 6]
labels = ["$e_1$", "$e_2$", "$e_3$", "$e_4$", "$e_5$"]
y_main = 0.0
node_radius = 0.24

ax.plot([xs[0] - 0.5, xs[-1] + 0.5], [y_main, y_main], color="#555555", lw=1.6, zorder=1)

for x, lab in zip(xs, labels):
    circle = plt.Circle((x, y_main), node_radius, facecolor="white",
                         edgecolor=NODE_EDGE, lw=1.6, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y_main, lab, ha="center", va="center", color=TEXT_COLOR,
            fontsize=11, fontweight="bold", zorder=4)

# "wait" self-loops under e1..e4: a dashed arc that returns to its own node.
decision_xs = xs[:-1]
for x in decision_xs:
    arc = patches.Arc((x, y_main - 0.30), width=0.5, height=0.5, angle=0,
                       theta1=200, theta2=340, color=WAIT_COLOR, ls=(0, (3, 2)),
                       lw=1.5, zorder=2)
    ax.add_patch(arc)
    ax.annotate(
        "", xy=(x + 0.235, y_main - 0.185), xytext=(x + 0.255, y_main - 0.29),
        arrowprops=dict(arrowstyle="-|>", color=WAIT_COLOR, lw=1.5, mutation_scale=10),
        zorder=2,
    )
    ax.text(x, y_main - 0.66, "wait", fontsize=9, fontweight="bold",
            color=WAIT_COLOR, ha="center", va="center")

# "intervene" and episode end at e4.
x_int = xs[3]
ax.annotate(
    "", xy=(x_int, y_main + 0.78), xytext=(x_int, y_main + node_radius + 0.05),
    arrowprops=dict(arrowstyle="-|>", color=INTERVENE_COLOR, lw=2.0, mutation_scale=13),
    zorder=2,
)
ax.text(x_int + 0.14, y_main + 0.44, "intervene", fontsize=9.5, fontweight="bold",
        color=INTERVENE_COLOR, ha="left", va="center")

badge_w, badge_h = 0.62, 0.30
badge = patches.FancyBboxPatch(
    (x_int - badge_w / 2, y_main + 0.78), badge_w, badge_h,
    boxstyle="round,pad=0.05,rounding_size=0.07",
    facecolor=INTERVENE_COLOR, edgecolor="none", zorder=4,
)
ax.add_patch(badge)
ax.text(x_int, y_main + 0.78 + badge_h / 2, "END", fontsize=9, fontweight="bold",
        color="white", ha="center", va="center", zorder=5)

# Callout: the 4-feature state that drives the decision, anchored at e2 (kept
# away from e4's intervene/END so the two annotations don't collide).
x_state = xs[1]
STATE_COLOR = "#4c72b0"
ax.annotate(
    "", xy=(x_state, y_main + node_radius + 0.04), xytext=(x_state, y_main + 0.40),
    arrowprops=dict(arrowstyle="-|>", color=STATE_COLOR, lw=1.2, mutation_scale=9),
    zorder=2,
)
state_box = patches.FancyBboxPatch(
    (x_state - 0.98, y_main + 0.42), 1.96, 0.52,
    boxstyle="round,pad=0.06,rounding_size=0.06",
    facecolor="#eef2f8", edgecolor=STATE_COLOR, lw=1.1, zorder=3,
)
ax.add_patch(state_box)
ax.text(
    x_state, y_main + 0.68,
    r"$s_t\,{=}\,$(progress, reliability, deviation,",
    fontsize=6.6, color=STATE_COLOR, ha="center", va="center", zorder=4,
)
ax.text(
    x_state, y_main + 0.55,
    r"free resources, effect estimates)",
    fontsize=6.6, color=STATE_COLOR, ha="center", va="center", zorder=4,
)

ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
ax.set_ylim(y_main - 0.95, y_main + 1.25)
ax.set_aspect("equal")
ax.axis("off")

fig.tight_layout()
fig.savefig(OUT / "fig1_mdp_timeline.pdf", bbox_inches="tight")
plt.close(fig)
print("fig1 done")

# ---------------------------------------------------------------------
# Fig 2: training curve of the BPIC2017 checkpoint (manifest annotation in
# the LaTeX caption). Rewards were scaled by `reward_scale` during training;
# the curve is shown in the unscaled units of the reward function.
# ---------------------------------------------------------------------
VARIANT = pools.DEFAULT_VARIANT
curve = pd.read_csv(paths.variant_artifact("BPIC2017", VARIANT, "_training_curve.csv"))
manifest = json.loads(paths.variant_artifact("BPIC2017", VARIANT, "_manifest.json").read_text())
scale = float(manifest.get("reward_scale", 1.0))
reward = curve["reward"] / scale
roll = reward.rolling(window=100, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.plot(curve["episode"], reward, color="#4c72b0", alpha=0.25, lw=0.6, label="episode return")
ax.plot(curve["episode"], roll, color="#c44e52", lw=1.8, label="rolling mean (window 100)")
ax.set_xlabel("Episode (one case)")
ax.set_ylabel("Return")
ax.legend(loc="upper left", fontsize=8, frameon=True)
fig.tight_layout()
fig.savefig(OUT / "fig2_training_curve.pdf", bbox_inches="tight")
plt.close(fig)
print("fig2 done, n_episodes =", manifest["n_episodes"], "mean_last_50 (unscaled) =",
      manifest["mean_episode_reward_last_50"] / scale, "state =", manifest.get("state_features"))

# ---------------------------------------------------------------------
# Fig 3 & 4: real explanation cards on BPIC2017 -- the median decision to
# act (IG on Delta Q) and the median decision to wait (IG on Delta Q_wait)
# of the Section 6 evaluation pool (pools.evaluation_pool).
# ---------------------------------------------------------------------
model = PPO.load(str(paths.variant_model("BPIC2017", VARIANT)), device="cpu")
policy = model.policy
states, _rows, FEATS = pools.evaluation_pool("BPIC2017", VARIANT)
reference = states.mean(axis=0)
print("reference (mean state):", dict(zip(FEATS, reference.round(3).tolist())))

margin_head = MarginHead(policy, intervene_action=1)
wait_head = WaitMarginHead(margin_head)
with torch.no_grad():
    dev = next(policy.parameters()).device
    m0 = margin_head(torch.from_numpy(states).to(dev)).cpu().numpy()
print(f"pool: {int((m0 > 0).sum())} act states, {int((m0 <= 0).sum())} wait states")

picks = pools.paper_card_indices(states, m0, _rows)  # median *correct* act and wait decisions
for tag, side in (("fig3", "act"), ("fig4", "wait")):
    if side not in picks:
        print(f"{tag}: no {side} state in the pool, skipped")
        continue
    idx = picks[side]
    s = states[idx : idx + 1]
    head, label = (margin_head, r"$\phi^{\Delta Q}$ (pushes toward acting now $\rightarrow$)") if side == "act" \
        else (wait_head, r"$\phi^{\Delta Q_{\mathrm{wait}}}$ (pulls toward waiting $\rightarrow$)")
    phi = integrated_gradients(head, s, reference, n_steps=128)[0]
    target = float(head(torch.from_numpy(s).to(dev)).detach().cpu().numpy()[0])
    share = np.abs(phi) / np.abs(phi).sum()
    print(f"{tag} ({side}): idx={idx} state={dict(zip(FEATS, s[0].round(3).tolist()))} target={target:.4f} "
          f"phi={dict(zip(FEATS, phi.round(4).tolist()))} share={dict(zip(FEATS, share.round(3).tolist()))} sum(phi)={phi.sum():.4f}")

    order = np.argsort(-np.abs(phi))
    feats_sorted = [FEATS[i] for i in order]
    phi_sorted = phi[order]
    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in phi_sorted]

    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    y = np.arange(len(feats_sorted))
    ax.barh(y, phi_sorted, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(feats_sorted, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(label)
    ax.margins(x=0.35)
    for yi, v in zip(y, phi_sorted):
        offset = 0.04 * max(abs(phi_sorted))
        ax.text(v + (offset if v >= 0 else -offset), yi, f"{v:.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, clip_on=False)
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}_explanation_card.pdf", bbox_inches="tight")
    plt.close(fig)

print("fig3/fig4 done")

# ---------------------------------------------------------------------
# Fig 3 (combined): the two cards side by side, one figure for the paper.
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.7))
for ax, side, title in zip(axes, ("act", "wait"), ("(a) decision to act", "(b) decision to wait")):
    if side not in picks:
        ax.axis("off"); continue
    idx = picks[side]
    s = states[idx : idx + 1]
    head, label = (margin_head, r"$\phi^{\Delta Q}$ (pushes toward acting now $\rightarrow$)") if side == "act" \
        else (wait_head, r"$\phi^{\Delta Q_{\mathrm{wait}}}$ (pulls toward waiting $\rightarrow$)")
    phi = integrated_gradients(head, s, reference, n_steps=128)[0]
    target = float(head(torch.from_numpy(s).to(dev)).detach().cpu().numpy()[0])
    order = np.argsort(-np.abs(phi))
    phi_sorted = phi[order]
    y = np.arange(len(order))
    ax.barh(y, phi_sorted, color=["#c44e52" if v > 0 else "#4c72b0" for v in phi_sorted])
    ax.set_yticks(y); ax.set_yticklabels([FEATS[i] for i in order], fontsize=8)
    ax.invert_yaxis(); ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(label, fontsize=8); ax.margins(x=0.35)
    ax.set_title(f"{title}, margin = {target:.2f}", fontsize=9)
    for yi, v in zip(y, phi_sorted):
        off = 0.04 * max(abs(phi_sorted))
        ax.text(v + (off if v >= 0 else -off), yi, f"{v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7, clip_on=False)
fig.tight_layout()
fig.savefig(OUT / "fig3_explanation_cards.pdf", bbox_inches="tight")
plt.close(fig)
print("fig3 (combined) done")
