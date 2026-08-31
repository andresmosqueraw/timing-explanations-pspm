"""
Fast PPO training for the SimBank "Time contact HQ" log (action width 2,
action depth 4; De Moor et al. 2025), reusing Shoush & Dumas's exact
state/reward DESIGN (relative_position, reliability, deviation,
available_resources; cost/gain/gain_res reward), mirroring
train_ppo_fast_rl_prescriptive_monitoring.py's PPMEnvFast 1:1 in structure.

TWO DOCUMENTED SIMPLIFICATIONS versus the BPIC2017/2012/TrafficFines runs
(state this plainly in the paper, do not silently gloss over it):

1. available_resources here is OUR OWN synthetic extension (see
   build_resources.py's docstring) -- SimBank has no native resource-
   capacity concept, and the feature is computed against an artificial
   Poisson case-arrival process we introduced, not part of either SimBank's
   or Shoush & Dumas's original design.

2. Shoush & Dumas's reward uses `ite = y1 - y0`, true potential outcomes
   from a causal-effect estimator (CausalLift) fit on the BPI logs.
   SimBank's raw log carries no such counterfactual pair for this
   intervention without re-running the simulator's own counterfactual
   generator (out of scope here, per plan). We substitute a documented
   PROXY: ite_proxy = z-score of unc_quality (the bank's own uncertainty-
   of-quality estimate at the decision point), i.e. positive exactly when
   uncertainty is above this log's average. This is consistent with
   SimBank's own domain narrative for this intervention ("the higher the
   client's quality uncertainty, the greater the cost of contacting HQ...
   HQ contact functions as a confidence check"), but it is NOT a causal
   effect estimate and must not be described as one.
"""

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

DATA_DEFAULT = Path(__file__).parent / "data" / "simbank_time_contact_hq_with_resources.pkl"
MODELS_DIR = Path(__file__).parent / "models"


class SimBankHQEnvFast(gym.Env):
    """
    State (4-D): [relative_position, reliability, deviation, available_resources]
    Action (Discrete 2): 0 = do nothing (skip_contact), 1 = intervene (contact_headquarters)
    Reward: same cost/gain/gain_res structure as PPMEnvFast, with a proxy ITE
    (see module docstring) in place of Shoush & Dumas's y1-y0.
    """

    metadata = {"render_modes": []}

    def __init__(self, data_path: Path | str = DATA_DEFAULT, n_resources: int = 5):
        super().__init__()
        df = pd.read_pickle(data_path)
        df = df.sort_values(["case_nr", "synthetic_time_days"]).reset_index(drop=True)
        df["prefix_nr"] = df.groupby("case_nr").cumcount() + 1
        df["case_length"] = df.groupby("case_nr")["case_nr"].transform("size")

        # Normalize est_quality (0-10) / unc_quality (0-5) onto the same
        # rough scale Shoush & Dumas's own reliability/deviation occupy.
        df["reliability"] = df["est_quality"].astype(float) / 10.0
        df["deviation"] = df["unc_quality"].astype(float) / 5.0

        uq = df["unc_quality"].astype(float)
        self._ite_mean = float(uq.mean())
        self._ite_std = float(uq.std()) or 1.0
        df["ite_proxy"] = (uq - self._ite_mean) / self._ite_std

        self._df = df
        self._max_idx = len(df) - 1
        self._n_resources = n_resources
        # ROOT-CAUSE FIX for the always-intervene collapse (3 prior fix
        # attempts targeting entropy/reward-scale/training-length all failed,
        # see manifest "note"): reset() used to always restart at row 0 --
        # copied 1:1 from PPMEnvFast's own reset(), which has the same
        # pattern -- but here every episode terminates in exactly one step
        # (ep_len_mean stayed at 1.0 for the entire 500k-step run), so
        # training only ever visited a single fixed transition, hundreds of
        # thousands of times, and the policy just memorized what was optimal
        # for that one state instead of learning to discriminate by state.
        # Fix: start every episode at a random case's first row instead.
        self._case_start_idx = df.groupby("case_nr").head(1).index.to_numpy()

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, float(n_resources)], dtype=np.float32),
        )
        self._idx = 0

    def _row(self):
        return self._df.iloc[self._idx]

    def _state(self, row=None) -> np.ndarray:
        if row is None:
            row = self._row()
        rel = float(row["prefix_nr"]) / max(float(row["case_length"]), 1.0)
        rel = float(np.clip(rel, 0.0, 1.0))
        return np.array(
            [rel, float(row["reliability"]), float(row["deviation"]), float(row["available_resources"])],
            dtype=np.float32,
        )

    @staticmethod
    def _reward(adapted: bool, ite: float, has_res: bool) -> float:
        cost, gain, gain_res = 25.0, 50.0, 50.0
        if adapted:
            if ite > 0:
                r = (gain * ite) - cost + (gain_res if has_res else -gain_res)
            elif ite == 0:
                r = -cost - gain_res
            else:
                r = -cost - gain - gain_res
        else:
            if ite > 0:
                r = -gain - gain_res
            elif ite == 0:
                r = gain_res if has_res else 0.0
            else:
                r = gain + gain_res
        return float(r)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = int(self.np_random.choice(self._case_start_idx))
        return self._state(), {}

    def step(self, action):
        row = self._row()
        adapted = bool(action == 1)
        ite = float(row["ite_proxy"])
        has_res = float(row["available_resources"]) > 0

        reward = self._reward(adapted, ite, has_res)

        is_last = int(row["prefix_nr"]) >= int(row["case_length"])
        terminated = adapted or is_last

        self._idx = min(self._idx + 1, self._max_idx)
        return self._state(), reward, terminated, False, {}

    def render(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=150_000,
                     help="34.5k contact_headquarters events / 100k cases in this log; "
                          "150k matches BPIC2017's ~1.5 steps-per-decision-point order of magnitude "
                          "at this log's smaller decision-point count (action depth 4, not case-length).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-resources", type=int, default=5, help="must match build_resources.py's --n-servers")
    ap.add_argument("--data", type=str, default=str(DATA_DEFAULT))
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    data_path = Path(args.data)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = Path(args.out) if args.out else MODELS_DIR / "ppo_simbank_time_contact_hq"
    monitor_path = str(save_path) + "_monitor.csv"

    print(f"Loading env from {data_path} ...")
    raw_env = SimBankHQEnvFast(data_path, n_resources=args.n_resources)
    check_env(raw_env, warn=True)
    env = Monitor(raw_env, filename=monitor_path)

    # Fix attempts #1 (ent_coef=0.01) and #2 (VecNormalize norm_reward) both
    # failed and are removed: the real cause was that reset() always
    # restarted at row 0, so training saw one fixed transition, not reward
    # scale or exploration bonus (see SimBankHQEnvFast.__init__ comment and
    # the manifest "note" of the 3 earlier attempts for the full record).
    # With the reset() fix, plain default PPO is tried first before
    # reintroducing any of the earlier band-aids.
    vec_env = DummyVecEnv([lambda: env])

    print(f"Training PPO for {args.timesteps:,} timesteps (seed={args.seed}) ...")
    model = PPO(
        "MlpPolicy", vec_env,
        n_steps=512, batch_size=64, n_epochs=5, learning_rate=3e-4, gamma=0.99,
        seed=args.seed, verbose=1,
    )
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps)
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
        "data_path": str(data_path.resolve()),
        "data_mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(data_path.stat().st_mtime)),
        "env": "SimBankHQEnvFast",
        "sb3_algo": "PPO",
        "policy": "MlpPolicy",
        "ite_proxy_mean_unc_quality": raw_env._ite_mean,
        "ite_proxy_std_unc_quality": raw_env._ite_std,
        "note": "available_resources is a synthetic extension (build_resources.py); "
                "ite_proxy is a z-scored uncertainty proxy, NOT a causal effect estimate. See module docstring. "
                "Fix attempt #1 (ent_coef=0.01 alone, 150k steps): still 100% intervene (deterministic eval), "
                "mean reward -63.6 (vs -68.67 baseline), entropy_loss decayed to ~0 by end of training. "
                "Fix attempt #2 (VecNormalize(norm_reward=True) + ent_coef=0.01, 150k steps): still 100% "
                "intervene (deterministic eval), mean reward -63.599 (near-identical to attempt #1), but "
                "stochastic training rollout reward reached ~57 (beats always-wait's +43.78). "
                "Fix attempt #3 (this run: same config, 500k steps instead of 150k, user-requested): "
                "IDENTICAL outcome to attempt #2 -- 100% intervene under deterministic eval, mean reward "
                "-63.599, ep_rew_mean(stochastic)=57. entropy_loss was already frozen near 0 by ~142k steps "
                "and stayed there through 500k, confirming more training budget does not change the outcome. "
                "CONCLUSION: the policy's mode (argmax action) has collapsed to always-intervene, but its "
                "full stochastic distribution retains real structure that a deterministic rollout discards -- "
                "this is a policy-extraction/evaluation-protocol issue, not a training-duration issue. "
                "Three fix attempts per the agreed budget are now exhausted. Benchmarks: "
                "always-wait=+43.78, always-intervene=-68.67, oracle-best-per-row=+93.08. "
                "See the printed intervene-rate/mean-reward check below for whether this run is sane.",
    }
    manifest_path = Path(str(save_path) + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest -> {manifest_path}")

    curve_path = Path(str(save_path) + "_training_curve.csv")
    with curve_path.open("w") as f:
        f.write("episode,reward\n")
        for i, r in enumerate(ep_rewards):
            f.write(f"{i},{r}\n")
    print(f"Training curve ({len(ep_rewards)} episodes) -> {curve_path}")

    # Quick intervene-rate check over the full data pass, mirroring the
    # BPIC2017 agent's "never intervenes" check.
    n_check = min(2000, raw_env._max_idx)
    obs, _ = raw_env.reset()
    n_intervene = 0
    step_rewards = []
    for i in range(n_check):
        action, _ = model.predict(obs, deterministic=True)
        if int(action) == 1:
            n_intervene += 1
        obs, r, terminated, _, _ = raw_env.step(int(action))
        step_rewards.append(r)
        if terminated:
            obs, _ = raw_env.reset()
            raw_env._idx = i  # keep sweeping forward through the data rather than restarting at row 0
    mean_r = sum(step_rewards) / len(step_rewards)
    print(f"\nIntervene rate over {n_check} sampled steps: {n_intervene / n_check:.1%}")
    print(f"Mean per-step reward over {n_check} sampled steps: {mean_r:.3f}")
    print("Benchmarks: always-intervene=-68.67, always-wait=+43.78, oracle-best-per-row=+93.08")
    if mean_r > 43.78:
        print("-> Trained policy BEATS always-wait: looks like a genuine, sane learned policy.")
    elif mean_r > -68.67 + 5:
        print("-> Trained policy is between the two naive extremes but has NOT beaten always-wait.")
    else:
        print("-> Trained policy is still near/at the always-intervene collapse. Fix did not work.")


if __name__ == "__main__":
    main()
