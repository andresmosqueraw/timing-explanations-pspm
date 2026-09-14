"""Fast PPO training for the Sepsis Cases - Event Log ("IV Antibiotics" ->
"Return ER"), reusing Shoush & Dumas's state/reward DESIGN exactly as
foreign/train_ppo_fast_rl_prescriptive_monitoring.py's "case"-episode recipe
does for BPIC ("cate" / "cate_retrained": one case per episode, cost/gain
reward, --reward-scale 0.01 --ent-coef 0.3), with ONE difference:

    NO available_resources term. Sepsis's log records no hospital-capacity
    variable at all (see pools.py's Sepsis section for the alternative
    considered and rejected -- synthesizing a Poisson-arrival capacity the
    way simbank_resources/build_resources.py does for SimBank -- and why).
    The state is therefore 5-D: [relative_position, reliability, deviation,
    Proba_if_Treated, Proba_if_Untreated], and the reward drops the
    gain_res/has_res terms accordingly.

This is the ONLY policy Sepsis ever gets: unlike BPIC there is no shipped
checkpoint trained on a different (incoherent) state to compare against, and
none should ever be added -- see build_sepsis_state.py's docstring and the
top-level task note about the SimBank shipped-vs-rebuilt failure mode this
repo must not repeat for Sepsis.

Usage:
    python sepsis_resources/train_ppo_sepsis.py \
        --timesteps 600000 --reward-scale 0.01 --ent-coef 0.3   # the BPIC recipe, 600k steps
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import paths  # noqa: E402

CSV_DEFAULT = paths.SEPSIS_STATE_CSV
MODELS_DIR = paths.VARIANT_MODELS
STATE_FEATURES = ["relative_position", "reliability", "deviation", "Proba_if_Treated", "Proba_if_Untreated"]


class SepsisEnv(gym.Env):
    """
    State (5-D): [relative_position, reliability, deviation, Proba_if_Treated, Proba_if_Untreated]
    Action (Discrete 2): 0 = do nothing, 1 = intervene (give IV Antibiotics)
    Reward: PPMEnvFast's cost/gain structure, minus the resource term (no
    available_resources here; see module docstring).

    Episode = one case (the paper's MDP, "episode=case" in the BPIC trainer):
    reset() draws a random case, steps walk its prefixes in order, the
    episode ends at the case's last event or at the intervention -- IV
    Antibiotics is a one-shot action in this reward design, matching how the
    BPIC/SimBank environments treat their own interventions.
    """

    metadata = {"render_modes": []}

    def __init__(self, csv_path: Path | str = CSV_DEFAULT, extra_features: tuple[str, ...] = ("Proba_if_Treated", "Proba_if_Untreated"),
                 reward_scale: float = 1.0):
        super().__init__()
        # keep_default_na=False: one real Sepsis case id is the literal
        # string "NA" (see pools.load_sepsis_state); without this it reads
        # back as NaN and silently merges that case's rows into one garbage group.
        df = pd.read_csv(csv_path, sep=";", dtype={"case_id": str}, keep_default_na=False, na_values=[])
        df = df.sort_values(["case_id", "prefix_nr"], kind="mergesort").reset_index(drop=True)
        self._df = df
        self._extra = tuple(extra_features)
        self._reward_scale = float(reward_scale)
        self._rng = np.random.default_rng(0)

        order = df.index.to_numpy()
        cid = df["case_id"].to_numpy()
        starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
        ends = np.r_[starts[1:], len(order)]
        self._case_rows = [order[s:e] for s, e in zip(starts, ends)]
        self._case_pos = 0
        self._case_seq = self._case_rows[0]

        self.action_space = spaces.Discrete(2)
        n_extra = len(self._extra)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0] + [0.0] * n_extra, dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0] + [1.0] * n_extra, dtype=np.float32),
        )
        self._idx = int(self._case_seq[0])

    def _row(self):
        return self._df.iloc[self._idx]

    def _state(self, row=None) -> np.ndarray:
        if row is None:
            row = self._row()
        rel = float(np.clip(float(row["prefix_nr"]) / max(float(row["progress_horizon"]), 1.0), 0.0, 1.0))
        return np.array([rel, float(row["reliability"]), float(row["deviation"])] + [float(row[c]) for c in self._extra], dtype=np.float32)

    @staticmethod
    def _reward(adapted: bool, ite: float) -> float:
        cost, gain = 25.0, 50.0
        if adapted:
            if ite > 0:
                r = (gain * ite) - cost
            elif ite == 0:
                r = -cost
            else:
                r = -cost - gain
        else:
            if ite > 0:
                r = -gain
            elif ite == 0:
                r = 0.0
            else:
                r = gain
        return float(r)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._case_seq = self._case_rows[int(self._rng.integers(0, len(self._case_rows)))]
        self._case_pos = 0
        self._idx = int(self._case_seq[0])
        return self._state(), {}

    def step(self, action):
        row = self._row()
        adapted = bool(action == 1)
        ite = float(row["y1"] - row["y0"])
        reward = self._reward(adapted, ite) * self._reward_scale

        self._case_pos += 1
        is_last = self._case_pos >= len(self._case_seq)
        terminated = adapted or is_last
        if not is_last:
            self._idx = int(self._case_seq[self._case_pos])

        return self._state(), reward, terminated, False, {}

    def render(self):
        pass


def always_wait_reward(df: pd.DataFrame) -> float:
    """Mean per-case reward of never intervening: the terminal reward of
    each case's own final (worst-information) prefix under the released
    reward's "wait" branch, averaged over cases -- the benchmark PPO must
    beat, mirroring the BPIC/SimBank scripts' own "always-wait" checks."""
    last = df.sort_values(["case_id", "prefix_nr"]).groupby("case_id").tail(1)
    ite = (last["y1"] - last["y0"]).to_numpy()
    return float(np.mean([SepsisEnv._reward(False, x) for x in ite]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=600_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", type=str, default=str(CSV_DEFAULT))
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--extra-features", nargs="*", default=["Proba_if_Treated", "Proba_if_Untreated"])
    ap.add_argument("--reward-scale", type=float, default=0.01)
    ap.add_argument("--ent-coef", type=float, default=0.3)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path} -- run build_sepsis_state.py first.", file=sys.stderr)
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = Path(args.out) if args.out else MODELS_DIR / "ppo_sepsis_cate"
    monitor_path = str(save_path) + "_monitor.csv"

    print(f"Loading env from {csv_path} ...")
    raw_env = SepsisEnv(csv_path, extra_features=tuple(args.extra_features), reward_scale=args.reward_scale)
    check_env(raw_env, warn=True)
    env = Monitor(raw_env, filename=monitor_path)

    print(f"Training PPO for {args.timesteps:,} timesteps (seed={args.seed}) ...")
    model = PPO(
        "MlpPolicy", env,
        n_steps=args.n_steps, batch_size=64, n_epochs=5, learning_rate=args.learning_rate, gamma=0.99,
        ent_coef=args.ent_coef, seed=args.seed, verbose=1,
    )
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps)
    elapsed = time.time() - t0

    model.save(str(save_path))
    print(f"\nModel saved -> {save_path}.zip")

    ep_rewards = list(env.get_episode_rewards())
    tail = ep_rewards[-50:] if ep_rewards else []

    # Deterministic sweep over every case's every prefix (the same CSV the
    # policy trained on -- Sepsis has no separate held-out pool; see
    # pools.SEPSIS_N_CASES) for the intervention-rate / gain-vs-always-wait
    # check every other trainer in this repo reports.
    df = raw_env._df
    n_intervene, n_rows, gain_sum, wait_sum = 0, 0, 0.0, 0.0
    for seq in raw_env._case_rows:
        obs, _ = raw_env.reset()
        raw_env._case_seq, raw_env._case_pos, raw_env._idx = seq, 0, int(seq[0])
        obs = raw_env._state()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            row = raw_env._row()
            ite = float(row["y1"] - row["y0"])
            gain_sum += SepsisEnv._reward(bool(action == 1), ite)
            wait_sum += SepsisEnv._reward(False, ite)
            n_rows += 1
            if int(action) == 1:
                n_intervene += 1
            obs, _, done, _, _ = raw_env.step(int(action))
    intervene_rate = n_intervene / max(n_rows, 1)
    always_wait = wait_sum / max(len(raw_env._case_rows), 1)
    policy_gain = gain_sum / max(len(raw_env._case_rows), 1)

    manifest = {
        "seed": args.seed, "total_timesteps": args.timesteps, "args": vars(args),
        "n_episodes": len(ep_rewards), "mean_episode_reward_last_50": (sum(tail) / len(tail)) if tail else None,
        "elapsed_seconds": elapsed, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "csv_path": str(csv_path.resolve()), "csv_mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(csv_path.stat().st_mtime)),
        "env": "SepsisEnv", "sb3_algo": "PPO", "policy": "MlpPolicy",
        "state_features": ["relative_position", "reliability", "deviation"] + list(args.extra_features),
        "episode": "case", "ent_coef": args.ent_coef, "reward_scale": args.reward_scale,
        "n_cases": len(raw_env._case_rows), "n_rows_swept": n_rows,
        "intervention_rate": intervene_rate,
        "mean_reward_per_case_policy": policy_gain,
        "mean_reward_per_case_always_wait": always_wait,
        "gain_vs_always_wait": policy_gain - always_wait,
        "note": "No available_resources term: Sepsis has no live-capacity column (see pools.py). "
                "Trained and swept on the same coherent CSV (build_sepsis_state.py's scored temporal test "
                "split, 263 cases) -- there is no separate shipped/rebuilt state to compare against for this log.",
    }
    manifest_path = Path(str(save_path) + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float))
    print(f"Manifest -> {manifest_path}")

    curve_path = Path(str(save_path) + "_training_curve.csv")
    with curve_path.open("w") as f:
        f.write("episode,reward\n")
        for i, r in enumerate(ep_rewards):
            f.write(f"{i},{r}\n")
    print(f"Training curve ({len(ep_rewards)} episodes) -> {curve_path}")

    print(f"\nIntervention rate over {n_rows} swept prefixes ({len(raw_env._case_rows)} cases): {intervene_rate:.1%}")
    print(f"Mean reward/case: policy={policy_gain:.2f}  always-wait={always_wait:.2f}  gain={policy_gain - always_wait:+.2f}")
    if policy_gain > always_wait:
        print("-> Policy BEATS always-wait.")
    else:
        print("-> Policy does NOT beat always-wait.")


if __name__ == "__main__":
    main()
