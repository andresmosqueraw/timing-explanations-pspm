import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
fig, ax = plt.subplots(figsize=(6.0, 2.4))
ax.set_xlim(-0.3, 6.3)
ax.set_ylim(-1.6, 1.3)
ax.axis("off")

xs = [0, 1.5, 3, 4.5, 6]
labels = ["$e_1$", "$e_2$", "$e_3$", "$e_4$", "$e_5$"]
for x, lab in zip(xs, labels):
    ax.scatter([x], [0], s=260, color="#4c72b0", zorder=3)
    ax.text(x, 0, lab, ha="center", va="center", color="white", fontsize=9, zorder=4)

for i in range(len(xs) - 1):
    ax.annotate(
        "", xy=(xs[i + 1] - 0.22, 0), xytext=(xs[i] + 0.22, 0),
        arrowprops=dict(arrowstyle="-", color="#888888", lw=1.2),
    )

decision_xs = xs[:-1]
for x in decision_xs:
    ax.annotate(
        "wait", xy=(x, -0.75), xytext=(x, -0.05),
        ha="center", va="top", fontsize=8, color="#c44e52",
        arrowprops=dict(arrowstyle="->", color="#c44e52", lw=1.1,
                         connectionstyle="arc3,rad=-0.35"),
    )
    ax.annotate(
        "", xy=(x + 0.55, -0.75), xytext=(x, -0.75),
        arrowprops=dict(arrowstyle="->", color="#c44e52", lw=1.0, ls=(0, (2, 2))),
    )

x_int = xs[3]
ax.annotate(
    "intervene", xy=(x_int, 0.95), xytext=(x_int, 0.05),
    ha="center", va="bottom", fontsize=8.5, color="#55a868", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#55a868", lw=1.4,
                     connectionstyle="arc3,rad=0.3"),
)
ax.scatter([x_int], [1.1], marker="s", s=170, color="#55a868", zorder=3)
ax.text(x_int, 1.1, "END", ha="center", va="center", color="white", fontsize=6.5, zorder=4)

ax.text(-0.1, -1.35, r"At every event, the policy re-asks: wait (dashed loop, stay open) or"
                      "\nintervene (solid arrow, episode ends). $a_0{=}$wait is a first-class action,"
                      "\nnot the absence of one. $e_5$ is simply the last event shown; the same"
                      "\nchoice recurs at every later event of a longer case.",
        fontsize=7.6, ha="left", va="top", style="italic")

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
ax.legend(loc="lower right", fontsize=8, frameon=True)
ax.set_title(
    f"seed={manifest['seed']}, total timesteps={manifest['total_timesteps']:,}, "
    f"{manifest['n_episodes']:,} episodes",
    fontsize=8.5,
)
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
