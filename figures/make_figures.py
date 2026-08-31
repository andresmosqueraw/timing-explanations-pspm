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

GEN = Path(__file__).resolve().parent.parent  # this repo's root (self-contained)
LIB = Path(
    "/home/andrew/Documents/docs/2-resolver-problema/process-mining/algorithms-explainability/libraries-prescriptive-process/01-rl-online"
)
OUT = Path("/home/andrew/Documents/docs/3-escribir-paper/conferencias/paper1/figures")
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(GEN))
from dual_level import MarginHead, WaitMarginHead, integrated_gradients  # noqa: E402
from run_dual_level_ppo import sample_rows  # noqa: E402

FEATS = ["relative_position", "reliability", "deviation", "available_resources"]

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
    r"$s_t\,{=}\,$(relative position, reliability,",
    fontsize=6.6, color=STATE_COLOR, ha="center", va="center", zorder=4,
)
ax.text(
    x_state, y_main + 0.55,
    r"deviation, available resources)",
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
# Fig 2: training curve (real data) with manifest annotation
# ---------------------------------------------------------------------
curve = pd.read_csv(GEN / "models/ppo_bpic2017_rl_prescriptive_monitoring_training_curve.csv")
manifest = json.loads(
    (GEN / "models/ppo_bpic2017_rl_prescriptive_monitoring_manifest.json").read_text()
)

roll = curve["reward"].rolling(window=100, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.plot(curve["episode"], curve["reward"], color="#4c72b0", alpha=0.25, lw=0.6,
        label="episode reward")
ax.plot(curve["episode"], roll, color="#c44e52", lw=1.8, label="rolling mean (window 100)")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.legend(loc="upper left", fontsize=8, frameon=True)
# No in-image title: the seed/timesteps/episode count are reported in the
# LaTeX caption instead, so the caption is self-contained (writing-guide
# rule: title belongs in the caption, not baked into the figure).
fig.tight_layout()
fig.savefig(OUT / "fig2_training_curve.pdf", bbox_inches="tight")
plt.close(fig)
print("fig2 done, n_episodes =", manifest["n_episodes"],
      "mean_last_50 =", manifest["mean_episode_reward_last_50"])

# ---------------------------------------------------------------------
# Fig 3 & 4: real explanation cards (IG on WaitMarginHead) for two states
# ---------------------------------------------------------------------
model = PPO.load(str(GEN / "models/ppo_bpic2017_rl_prescriptive_monitoring.zip"), device="cpu")
policy = model.policy

# Rebuild the exact same 500-state sample (seed 123) the JSON artifacts used.
csv = LIB / "RL-prescriptive-monitoring/rl/data/ready_to_use_adaptive_bpic2017.csv"
df = pd.read_csv(csv, sep=";")
rows = sample_rows(df, "case_id")
rel = (rows["prefix_nr"] / rows["case_length"].clip(lower=1)).clip(0, 1)
resources = np.arange(len(rows)) % 4
states = np.stack(
    [rel.to_numpy(), rows["reliability"].to_numpy(), rows["deviation"].to_numpy(), resources],
    axis=1,
).astype(np.float32)
reference = states.mean(axis=0)
print("reference (mean state):", dict(zip(FEATS, reference.round(3).tolist())))

margin_head = MarginHead(policy, intervene_action=1)
wait_head = WaitMarginHead(margin_head)

with torch.no_grad():
    dev = next(policy.parameters()).device
    m0 = margin_head(torch.from_numpy(states).to(dev)).cpu().numpy()
assert (m0 < 0).all(), "expected every sampled state to be a wait state"

# Pick two illustrative states with different available_resources profiles:
# one scarce-resource case, one relatively resource-rich case.
idx_scarce = int(np.argsort(states[:, 3])[5])   # low resources, not the extreme outlier
idx_rich = int(np.argsort(-states[:, 3])[5])    # high resources, not the extreme outlier

for tag, idx in [("fig3", idx_scarce), ("fig4", idx_rich)]:
    s = states[idx : idx + 1]
    phi = integrated_gradients(wait_head, s, reference, n_steps=128)[0]
    dq_wait = float(-margin_head(torch.from_numpy(s).to(dev)).detach().cpu().numpy()[0])
    print(f"{tag}: idx={idx} state={dict(zip(FEATS, s[0].round(3).tolist()))} "
          f"dQ_wait={dq_wait:.4f} phi={dict(zip(FEATS, phi.round(4).tolist()))} "
          f"sum(phi)={phi.sum():.4f}")

    order = np.argsort(-np.abs(phi))
    feats_sorted = [FEATS[i] for i in order]
    phi_sorted = phi[order]
    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in phi_sorted]

    fig, ax = plt.subplots(figsize=(5.4, 2.3))
    y = np.arange(len(feats_sorted))
    ax.barh(y, phi_sorted, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(feats_sorted, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(r"$\phi^{\Delta Q_{\mathrm{wait}}}$ (pulls toward waiting $\rightarrow$)")
    # Generous x-margin so the value labels placed just past each bar's tip
    # never reach the left/right axes edge, where the y-tick labels live --
    # on a tight bbox that region can otherwise collide with the tip label
    # of the longest bar.
    ax.margins(x=0.35)
    for yi, v in zip(y, phi_sorted):
        offset = 0.04 * max(abs(phi_sorted))
        ax.text(v + (offset if v >= 0 else -offset),
                 yi, f"{v:.3f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=8, clip_on=False)
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}_explanation_card.pdf", bbox_inches="tight")
    plt.close(fig)

print("fig3/fig4 done")
