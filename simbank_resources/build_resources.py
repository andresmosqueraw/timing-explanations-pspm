"""
Build a synthetic `available_resources` feature for SimBank's "Time contact
HQ" log (action width 2, action depth 4; De Moor et al. 2025).

WHY THIS EXISTS (read before trusting the numbers)
----------------------------------------------------
The as-generated log (`tdqn/data/raw/loan_log_[_time_contact_HQ_]_100000_
train_normal`) places every case on its OWN, non-overlapping slice of the
timestamp axis: case i+1's earliest event always starts strictly after case
i's latest event. Verified directly on this file: 0 of 99,999 consecutive
case pairs overlap. This is a property of how SimBank's sequential,
single-simpy-process case generator advances `simulation_start` after each
case (see SimBank/SimBank/simulation.py) -- it is not a bug, just means the
raw timestamps encode "cases generated one after another", not "cases
arriving concurrently at a bank". A resource-capacity constraint computed
directly on these timestamps would find zero contention, ever: every
contact_headquarters event would always see all N servers free.

De Moor et al.'s SimBank does not model staff/resource capacity for HQ
contact at all (confirmed: no simpy.Resource, no capacity/queue notion
anywhere in SimBank/SimBank/*.py). Shoush & Dumas's `available_resources`
feature, which the policy we explain (Paper 1) was designed against, has no
counterpart in this log. This script is OUR OWN extension, not part of
either the original SimBank or Shoush & Dumas design, built to give this
log a comparable resource-scarcity dimension so the same PPO methodology
can be applied to it. It must be described as such in the paper, not
presented as if it were part of SimBank's own design.

METHOD
------
1. Give each case a synthetic ARRIVAL time via a Poisson process at rate
   `--arrival-rate` cases/day (cases in the raw log have no arrival time of
   their own -- only relative, case-internal `elapsed_time`, in days).
2. Every event's SYNTHETIC absolute time = case arrival time + elapsed_time.
   This is what creates genuine concurrency between cases; the original
   `timestamp` column is never modified or read again after this step.
3. Extract contact_headquarters events on this synthetic timeline, and run
   a greedy earliest-free-server assignment against N servers (`--n-servers`),
   each busy for the activity's fixed duration (6 days, from
   `times_dic["contact_headquarters"] = 6 * 86400` seconds in
   SimBank/SimBank/activity_execution.py).
4. For every row of the full log, `available_resources` = N minus the
   number of servers whose busy interval contains that row's synthetic
   time. No other column is modified -- this is a purely descriptive
   congestion signal, not a re-simulation of delayed downstream activities
   (a case's own outcome/timestamps are exactly as SimBank generated them;
   only the new column is added).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(
    "/home/andrew/Documents/docs/2-resolver-problema/process-mining/algorithms-explainability"
    "/tdqn/data/raw/loan_log_[_time_contact_HQ_]_100000_train_normal"
)
OUT_DIR = Path(__file__).parent / "data"
HQ_DURATION_DAYS = 6.0  # SimBank/SimBank/activity_execution.py: times_dic["contact_headquarters"] = 6 * 86400 sec


def synthetic_schedule(df: pd.DataFrame, arrival_rate: float, seed: int) -> pd.Series:
    """Poisson-process case arrivals (rate = cases/day); return per-row synthetic time (days, float)."""
    rng = np.random.default_rng(seed)
    case_ids = df["case_nr"].unique()
    case_ids.sort()
    n_cases = len(case_ids)
    interarrival = rng.exponential(scale=1.0 / arrival_rate, size=n_cases)
    arrival_time = np.cumsum(interarrival)
    arrival_by_case = pd.Series(arrival_time, index=case_ids)

    # elapsed_time is already in days (case-relative clock, verified against
    # this case's own timestamp deltas: e.g. case 0's first two events are
    # 2h24m apart, elapsed_time step 0.1 day = 2.4h -- consistent).
    synth = df["case_nr"].map(arrival_by_case).to_numpy() + df["elapsed_time"].to_numpy()
    return pd.Series(synth, index=df.index, name="synthetic_time_days")


def assign_servers(hq_times: np.ndarray, n_servers: int):
    """Greedy earliest-free-server assignment. Returns array of busy-interval starts, one per server slot used
    (a server can be reused sequentially), as a list of (start, end) per server "lane" concatenated."""
    order = np.argsort(hq_times)
    free_at = np.zeros(n_servers)  # next time each server is free
    server_of_event = np.empty(len(hq_times), dtype=int)
    busy_intervals = []  # (start, end) for every assignment, across all servers
    for idx in order:
        t = hq_times[idx]
        s = int(np.argmin(free_at))
        server_of_event[idx] = s
        start = max(t, free_at[s])  # if somehow queued (shouldn't happen given greedy-earliest logic), start later
        end = start + HQ_DURATION_DAYS
        free_at[s] = end
        busy_intervals.append((start, end))
    return np.array(busy_intervals)  # shape (n_hq_events, 2), sorted by original hq_times order via `order`... see note below


def compute_available_resources(all_times: np.ndarray, busy_intervals: np.ndarray, n_servers: int) -> np.ndarray:
    """For each row's synthetic time, count how many of the busy_intervals contain it, return N - that count."""
    starts = np.sort(busy_intervals[:, 0])
    ends = np.sort(busy_intervals[:, 1])
    # number of intervals that have started by t, minus number that have ended by t
    started = np.searchsorted(starts, all_times, side="right")
    ended = np.searchsorted(ends, all_times, side="right")
    busy_count = started - ended
    return n_servers - busy_count


def report_utilization(hq_times: np.ndarray, n_servers: int) -> dict:
    busy_intervals = assign_servers(hq_times, n_servers)
    total_busy_time = len(hq_times) * HQ_DURATION_DAYS
    horizon = hq_times.max() - hq_times.min() + HQ_DURATION_DAYS
    utilization = total_busy_time / (n_servers * horizon)
    return {"n_servers": n_servers, "utilization": utilization, "busy_intervals": busy_intervals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrival-rate", type=float, default=1.446, help="cases/day, Poisson rate (see docstring)")
    ap.add_argument("--n-servers", type=int, default=None, help="fix N; if omitted, sweep and report")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(OUT_DIR / "simbank_time_contact_hq_with_resources.pkl"))
    args = ap.parse_args()

    print(f"Loading {RAW_PATH} ...")
    df = pd.read_pickle(RAW_PATH)
    print(f"  {len(df):,} rows, {df['case_nr'].nunique():,} cases")

    synth = synthetic_schedule(df, args.arrival_rate, args.seed)
    df = df.assign(synthetic_time_days=synth)

    hq_mask = df["activity"] == "contact_headquarters"
    hq_times = df.loc[hq_mask, "synthetic_time_days"].to_numpy()
    print(f"  {hq_mask.sum():,} contact_headquarters events on synthetic timeline")
    print(f"  synthetic horizon: {df['synthetic_time_days'].max() - df['synthetic_time_days'].min():.1f} days"
          f" (arrival_rate={args.arrival_rate}/day)")

    if args.n_servers is None:
        print("\nSweeping N (no --n-servers given):")
        for n in [2, 3, 4, 5, 6, 8, 10, 15, 20]:
            r = report_utilization(hq_times, n)
            print(f"  N={n:3d}  utilization={r['utilization']:.1%}")
        print("\nRe-run with --n-servers <N> once you've picked one.")
        return

    r = report_utilization(hq_times, args.n_servers)
    print(f"\nChosen N={args.n_servers}, utilization={r['utilization']:.1%}")

    avail = compute_available_resources(
        df["synthetic_time_days"].to_numpy(), r["busy_intervals"], args.n_servers
    )
    df["available_resources"] = avail
    print(f"  available_resources: min={avail.min()}, max={avail.max()}, mean={avail.mean():.2f}")
    print(f"  fraction of rows with 0 available: {(avail == 0).mean():.1%}")
    print(f"  fraction of rows with N available (fully free): {(avail == args.n_servers).mean():.1%}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.out)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
