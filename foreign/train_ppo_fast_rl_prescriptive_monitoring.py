"""
Fast PPO training for RL-prescriptive-monitoring using BPIC 2017.

Bypasses the original envManager_v2 (argv/hardcoded paths) and
baselineEnv (TensorFlow dependency) by reimplementing the env logic
directly from the CSV. Faithful to the original reward and state design.

Usage:
    python train_ppo_fast.py [--timesteps 300000] [--seed 42] [--resources 3]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

# Retrained here from Shoush & Dumas's own preprocessed CSV (see README "Data").
# BPIC2012 was trained with --csv data/ready_to_use_adaptive_bpic2012.csv and
# --out models/ppo_bpic2012_rl_prescriptive_monitoring.
CSV_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "ready_to_use_adaptive_bpic2017.csv"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "models"


class PPMEnvFast(gym.Env):
    """
    Minimal Gym env for RL-prescriptive-monitoring.

    State (4-D): [relative_position, reliability, deviation, n_resources]
    Action (Discrete 2): 0 = do nothing, 1 = intervene
    Reward: replicates compute_reward() from state_with_temp_costReward_withoutPreds.py
    """

    metadata = {"render_modes": []}

    def __init__(self, csv_path: Path | str = CSV_DEFAULT, resources: int = 3,
                 extra_features: tuple[str, ...] = (), random_start: bool = False,
                 episode: str = "stream", reward_scale: float = 1.0, random_resources: bool = False):
        """``extra_features``, ``random_start``, ``episode="case"`` and
        ``random_resources`` are NOT part of the released design (all default
        off, which reproduces it):

        * ``extra_features``: CSV columns appended to the 4-feature state, e.g.
          the causal model's counterfactual outcome probabilities
          ``Proba_if_Treated``/``Proba_if_Untreated``. Under the released
          state, P(ite > 0 | s) never exceeds ~0.3 in any observable cell
          while the reward's per-row break-even is ~0.6, so the state cannot
          tell a treatable case from an untreatable one; the released
          *paper* agent sees treatment-effect estimates the released *code*
          does not expose, and this option restores that information.
        * ``random_start``: start each episode at a uniformly random row
          instead of row 0. With the released reset() every episode replays
          the CSV from its first row.
        * ``episode``: ``"stream"`` (released; envManager_v2) walks the
          timestamp-sorted log across all cases and ends the episode when
          *any* case finishes or the agent intervenes. Because not
          intervening pays +50/+100 on every row whose effect is <= 0 (81-97%
          of rows) and intervening ends the stream, intervening forfeits
          dozens of positive rewards from unrelated cases and is never
          optimal, whatever the state says -- the treatment-effect features
          alone do not make the agent intervene. ``"case"`` makes the episode
          one case, as the paper's MDP states it: reset() draws a random
          case, steps walk its events in order, and the episode ends at the
          case's last event or at the intervention. Waiting on a case whose
          effect is positive then costs -100 per remaining event, and
          intervening on it pays +75 once, so a state that can tell such
          cases apart is rewarded for acting on them.
        * ``random_resources``: draw the episode's free resources uniformly
          from ``0..resources`` at reset instead of always starting at
          ``resources``. In the released design a resource is only consumed
          by the intervention, which ends the episode, so the agent never
          observes anything but ``resources`` and any attribution to
          ``available_resources`` on a pool that varies it is extrapolation.
          The evaluation pool cycles it through ``0..resources``
          (pools.cycled_resources); this option trains on the same range.
        """
        super().__init__()
        df = pd.read_csv(csv_path, sep=";")
        # sort event-by-event as original envManager_v2 does
        df = df.sort_values(["orig_timestamp", "prefix_nr"]).reset_index(drop=True)
        self._df = df
        self._max_idx = len(df) - 1
        self._resources_init = resources
        self._extra = tuple(extra_features)
        self._random_start = random_start
        self._episode = episode
        # reward_scale: multiplies every reward the agent sees (the released
        # rewards are +-50..125 per step; with them PPO's actor saturates to a
        # deterministic never-intervene within a few thousand steps -- approx_kl
        # and entropy both reach exactly 0 -- and no entropy bonus of a
        # sensible size can compete). Evaluation scripts never use it.
        self._reward_scale = float(reward_scale)
        self._random_resources = bool(random_resources)
        self._rng = np.random.default_rng(0)
        if episode == "case":
            # per-case row lists, in each case's own event order
            order = df.sort_values(["case_id", "prefix_nr"]).index.to_numpy()
            cid = df.loc[order, "case_id"].to_numpy()
            starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
            ends = np.r_[starts[1:], len(order)]
            self._case_rows = [order[s:e] for s, e in zip(starts, ends)]
            self._case_pos = 0
            self._case_seq = self._case_rows[0]
        elif episode != "stream":
            raise ValueError(episode)

        self.action_space = spaces.Discrete(2)
        n_extra = len(self._extra)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1e6, -1e6, 0.0] + [-1e6] * n_extra, dtype=np.float32),
            high=np.array([1.0, 1e6, 1e6, float(resources)] + [1e6] * n_extra, dtype=np.float32),
        )
        self._idx = 0
        self._nr_res = list(range(1, resources + 1))

    # ── helpers ──────────────────────────────────────────────────────────────

    def _row(self):
        return self._df.iloc[self._idx]

    def _state(self, row=None) -> np.ndarray:
        if row is None:
            row = self._row()
        # progress_horizon (coherent CSVs, known in advance) or the released case_length
        denom = row["progress_horizon"] if "progress_horizon" in row.index else row["case_length"]
        rel = float(row["prefix_nr"]) / max(float(denom), 1.0)
        rel = float(np.clip(rel, 0.0, 1.0))
        base = [rel, float(row["reliability"]), float(row["deviation"]), float(len(self._nr_res))]
        return np.array(base + [float(row[c]) for c in self._extra], dtype=np.float32)

    @staticmethod
    def _reward(adapted: bool, ite: float, nr_res: list) -> float:
        cost, gain, gain_res = 25.0, 50.0, 50.0
        has_res = len(nr_res) > 0
        if adapted:
            if ite > 0:
                r = (gain * ite) - cost + (gain_res if has_res else -gain_res)
            elif ite == 0:
                r = -cost - gain_res
            else:
                r = -cost - gain - gain_res
        else:
            if ite > 0:
                r = -gain - (gain_res if has_res else gain_res)
            elif ite == 0:
                r = gain_res if has_res else 0.0
            else:
                r = gain + (gain_res if has_res else gain_res)
        return float(r)

    # ── gym API ───────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._episode == "case":
            self._case_seq = self._case_rows[int(self._rng.integers(0, len(self._case_rows)))]
            self._case_pos = 0
            self._idx = int(self._case_seq[0])
        else:
            self._idx = int(self._rng.integers(0, self._max_idx + 1)) if self._random_start else 0
        n_res = int(self._rng.integers(0, self._resources_init + 1)) if self._random_resources else self._resources_init
        self._nr_res = list(range(1, n_res + 1))
        return self._state(), {}

    def step(self, action):
        row = self._row()
        adapted = bool(action == 1)

        y0, y1 = float(row["y0"]), float(row["y1"])
        ite = y1 - y0

        if adapted and self._nr_res:
            self._nr_res.pop(0)

        reward = self._reward(adapted, ite, self._nr_res) * self._reward_scale

        if self._episode == "case":
            # episode = this case: ends at its last event or at the intervention
            self._case_pos += 1
            is_last = self._case_pos >= len(self._case_seq)
            terminated = adapted or is_last
            if not is_last:
                self._idx = int(self._case_seq[self._case_pos])
        else:
            # released: episode ends when a case is treated OR the row is the
            # last event of *its* case (any case in the stream)
            is_last = int(row["prefix_nr"]) >= int(row["case_length"])
            terminated = adapted or is_last
            # advance index (wrap around to enable continuous collection)
            self._idx = min(self._idx + 1, self._max_idx)

        obs = self._state()
        return obs, reward, terminated, False, {}

    def render(self):
        pass


class _SaveAt:
    """Save the model when training passes each of ``steps`` (one run gives
    every checkpoint of the validation sweep)."""

    def __new__(cls, steps, base):
        from stable_baselines3.common.callbacks import BaseCallback

        class SaveAt(BaseCallback):
            def __init__(self):
                super().__init__()
                self.todo = sorted(int(s) for s in steps)

            def _on_step(self) -> bool:
                while self.todo and self.num_timesteps >= self.todo[0]:
                    self.model.save(f"{base}_{self.todo.pop(0) // 1000}k")
                return True

        return SaveAt()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000,
        help="Total PPO training timesteps (default 300 000, per README provenance claim)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resources", type=int, default=3)
    parser.add_argument("--csv", type=str, default=str(CSV_DEFAULT))
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output basename (no extension); default models/ppo_bpic2017_rl_prescriptive_monitoring",
    )
    parser.add_argument("--extra-features", nargs="*", default=[],
                        help="CSV columns appended to the state (not in the released design; see PPMEnvFast)")
    parser.add_argument("--random-start", action="store_true",
                        help="start episodes at a random row (not in the released design; see PPMEnvFast)")
    parser.add_argument("--episode", choices=["stream", "case"], default="stream",
                        help="'stream' = released episode structure; 'case' = one case per episode (see PPMEnvFast)")
    parser.add_argument("--random-resources", action="store_true",
                        help="draw the initial free resources uniformly from 0..--resources at each reset (not in the released design; see PPMEnvFast)")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="PPO entropy coefficient (SB3 default 0.0). With 0 the actor collapses to never-intervene "
                             "within the first few thousand steps -- random interventions are punished on the 81-97%% of "
                             "rows whose effect is <= 0 before the state can tell those rows apart -- and never explores "
                             "again, even when the critic has learned that positive-effect states are worth less (V=28 vs 227 "
                             "on BPIC2012 with the cate features).")
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="multiply the training rewards (see PPMEnvFast); evaluation is always on the unscaled reward")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--save-at", nargs="*", type=int, default=[], help="also save checkpoints <out>_<k>k at these timesteps")
    parser.add_argument("--threads", type=int, default=0, help="torch threads (0 = torch default)")
    parser.add_argument("--n-steps", type=int, default=512)
    args = parser.parse_args()
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading env from {csv_path} ...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = Path(args.out) if args.out else RESULTS_DIR / "ppo_bpic2017_rl_prescriptive_monitoring"
    monitor_path = str(save_path) + "_monitor.csv"

    raw_env = PPMEnvFast(csv_path, resources=args.resources,
                         extra_features=tuple(args.extra_features), random_start=args.random_start,
                         episode=args.episode, reward_scale=args.reward_scale, random_resources=args.random_resources)
    check_env(raw_env, warn=True)
    env = Monitor(raw_env, filename=monitor_path)

    print(f"Training PPO for {args.timesteps:,} timesteps (seed={args.seed}) ...")
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=args.n_steps,
        batch_size=64,
        n_epochs=5,
        learning_rate=args.learning_rate,
        gamma=0.99,
        ent_coef=args.ent_coef,
        seed=args.seed,
        verbose=1,
    )
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=_SaveAt(args.save_at, save_path) if args.save_at else None)
    elapsed = time.time() - t0

    model.save(str(save_path))
    print(f"\nModel saved -> {save_path}.zip")

    ep_rewards = list(env.get_episode_rewards())
    tail = ep_rewards[-50:] if ep_rewards else []
    manifest = {
        "seed": args.seed,
        "total_timesteps": args.timesteps,
        "args": vars(args),
        "n_episodes": len(ep_rewards),
        "mean_episode_reward_last_50": (sum(tail) / len(tail)) if tail else None,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "csv_path": str(csv_path.resolve()),
        "csv_mtime": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(csv_path.stat().st_mtime)
        ),
        "env": "PPMEnvFast",
        "sb3_algo": "PPO",
        "policy": "MlpPolicy",
        "state_features": ["relative_position", "reliability", "deviation", "available_resources"] + list(args.extra_features),
        "random_start": bool(args.random_start),
        "random_resources": bool(args.random_resources),
        "episode": args.episode,
        "ent_coef": args.ent_coef,
        "reward_scale": args.reward_scale,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
    }
    manifest_path = Path(str(save_path) + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"   Manifest → {manifest_path}")

    curve_path = Path(str(save_path) + "_training_curve.csv")
    with curve_path.open("w") as f:
        f.write("episode,reward\n")
        for i, r in enumerate(ep_rewards):
            f.write(f"{i},{r}\n")
    print(f"   Training curve ({len(ep_rewards)} episodes) → {curve_path}")


if __name__ == "__main__":
    main()
