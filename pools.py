"""The state pools every script in this repository evaluates on.

One definition, so the explanation cards and Table "card" (paper Section 6),
the deletion test (Section 7) and the gain recomputation (Section 5.3) all
read the same states, built the same way:

* **Evaluation pool** (``bpic_pool`` / ``simbank_pool``): one random prefix
  per case, ``N_CASES`` cases, seed ``POOL_SEED``. This is the pool the
  Section 6 cards were drawn from; Section 7 runs the deletion test on the
  same pool.

* **Full pool** (``bpic_full`` / ``simbank_full``): every decision point of
  the log, for the historical-vs-policy gain table.

``available_resources`` on the BPIC logs. Shoush & Dumas's logs record no
live capacity, and their environment (mirrored in
``foreign/train_ppo_fast_rl_prescriptive_monitoring.py``) only ever shows the
policy the initial pool of ``N_RESOURCES`` free resources: a resource is
consumed by the intervention, and the intervention ends the episode. Two
conventions therefore coexist, and each pool states which one it uses:

* the **evaluation pool** cycles ``available_resources`` through
  ``0, 1, ..., N_RESOURCES`` across its rows, so that the feature actually
  varies and can receive (and be tested for) an attribution;
* the **full pool** holds it at ``N_RESOURCES``, the value the policy was
  trained under, so the gain recomputation matches the training condition.

SimBank's ``available_resources`` is a real column of the resource-augmented
log (``simbank_resources/build_resources.py``) and is used as recorded; the
``cate`` variant reads the effect-augmented pkl (``paths.simbank_pkl``), whose
``Proba_if_*`` / ``y1`` / ``y0`` come from the retrained estimator.

SimBank decision points. The SimBank checkpoint was trained stepping through
every event of a case, exactly as the BPIC checkpoints were (see
``simbank_resources/train_ppo_simbank.py``), so by default both SimBank pools
treat every event as a decision point (``rows="all"``). The log also records
exactly one contact decision per case, as a ``contact_headquarters`` or
``skip_contact`` event (``DECISION_ACTIVITIES``); ``rows="decision"`` restricts
the pools to those rows. That variant is kept as a diagnostic only: on every
one of those 93,337 rows the simulator's quality uncertainty is already 0
(it is resolved by the preceding email/validate loop), so the uncertainty
proxy that stands in for the treatment effect is a single negative constant
there and cannot discriminate anything (compute_gain_table.py records the
result under ``simbank_decision_rows``).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import paths

sys.path.insert(0, str(paths.REPO / "simbank_resources"))
from train_ppo_simbank import SimBankHQEnvFast  # noqa: E402

FEATS = ["relative_position", "reliability", "deviation", "available_resources"]
# Policy variants trained with extra state features appended to FEATS (see
# foreign/train_ppo_fast_rl_prescriptive_monitoring.py --extra-features /
# --episode). Both add the causal model's counterfactual outcome
# probabilities, the information the released 4-feature state lacks to tell
# a treatable case from an untreatable one:
#   "cate"        -- with one case per episode (--episode case), the MDP the
#                    paper states, rewards scaled by 0.01 and ent_coef 0.3 so
#                    the actor keeps exploring while the critic learns; this
#                    is the variant that intervenes.
#   "cate_stream" -- with the released stream episodes (--random-start); kept
#                    as the negative result: intervening ends the stream and
#                    forfeits the positive per-row rewards of unrelated cases,
#                    so the agent never intervenes however informative the state.
#   "cate_noent"  -- per-case episodes but SB3's default ent_coef=0: the
#                    actor collapses to never-intervene in the first few
#                    thousand steps and never explores again, although its
#                    critic learns that positive-effect states are worth less.
VARIANTS: dict[str, list[str]] = {
    "released": [],
    "cate": ["Proba_if_Treated", "Proba_if_Untreated"],
    "cate_stream": ["Proba_if_Treated", "Proba_if_Untreated"],
    "cate_noent": ["Proba_if_Treated", "Proba_if_Untreated"],
    # per-case, ent_coef=0.02, unscaled rewards: same collapse (approx_kl and
    # entropy reach exactly 0); the +-100 rewards swamp any entropy bonus
    "cate_ent002": ["Proba_if_Treated", "Proba_if_Untreated"],
    # the "cate" recipe trained on the *coherent* state: the prepared log's
    # test split scored by the retrained risk and effect models
    # (build_retrained_state.py, paths.retrained_csv), so the two lower
    # levels explain exactly the numbers the policy reads. The paper's policy.
    "cate_retrained": ["Proba_if_Treated", "Proba_if_Untreated"],
    # the same recipe on the coherent state *without* the effect features:
    # the released four-feature state, so the risk coordinates are the only
    # lower-level outputs the policy reads (the composition with a live risk
    # channel). Always-wait is optimal for it, so it only has a wait side.
    "risk_retrained": [],
}
COHERENT_VARIANTS = ("cate_retrained", "risk_retrained")
# Column holding the historically recorded action in each BPIC CSV
TREATMENT_COL = {"released": {"BPIC2012": "Treatment", "BPIC2017": "treatment"}}


def treatment_col(log: str, variant: str) -> str:
    return "treatment" if variant in COHERENT_VARIANTS else TREATMENT_COL["released"][log]


# The policy the paper explains: the "cate" variant on all three logs. On
# SimBank the two extra columns come from the retrained two-model estimator
# (effect_model.py, simbank_resources/add_effect_features.py) and "released"
# is the earlier 4-feature checkpoint whose effect signal was the uncertainty
# proxy.
DEFAULT_VARIANT = "cate_retrained"
LOGS = ("BPIC2012", "BPIC2017", "SimBank")


def extra_cols(log: str, variant: str = DEFAULT_VARIANT) -> list[str]:
    return list(VARIANTS[variant])


def feature_names(log: str = "BPIC2017", variant: str = DEFAULT_VARIANT) -> list[str]:
    return FEATS + extra_cols(log, variant)


def evaluation_pool(log: str, variant: str = DEFAULT_VARIANT):
    """(states, rows, feature_names) of the Section 6/7 pool for ``log``."""
    if log in ("BPIC2012", "BPIC2017"):
        states, rows = bpic_pool(paths.bpic_csv(log, variant), variant=variant)
    elif log == "SimBank":
        states, rows = simbank_pool(paths.simbank_pkl(variant), variant=variant)
    else:
        raise ValueError(log)
    return states, rows, feature_names(log, variant)


POOL_SEED = 123
N_CASES = 500
N_RESOURCES = 3  # Shoush & Dumas's initial resource pool, as trained
DECISION_ACTIVITIES = ("skip_contact", "contact_headquarters")
SIMBANK_TREATMENT_ACTIVITY = "contact_headquarters"

# ---------------------------------------------------------------------------
# BPIC2012 / BPIC2017 (Shoush & Dumas's preprocessed CSVs)
# ---------------------------------------------------------------------------


def load_bpic(csv_path) -> pd.DataFrame:
    """The CSV in its own row order (the order the evaluation sample is drawn in)."""
    return pd.read_csv(csv_path, sep=";")


def sample_one_prefix_per_case(df: pd.DataFrame, case_col: str, n_cases: int = N_CASES, seed: int = POOL_SEED) -> pd.DataFrame:
    picked = df.groupby(case_col, sort=False).sample(n=1, random_state=seed)
    if len(picked) > n_cases:
        picked = picked.sample(n=n_cases, random_state=seed)
    return picked.reset_index(drop=True)


def oracle_acts(rows: pd.DataFrame) -> np.ndarray:
    """Whether intervening pays at each pool row under the released reward:
    the row's estimated treatment effect is positive (y1 - y0 > 0 on the
    BPIC logs, the uncertainty proxy on SimBank)."""
    if "y1" in rows.columns:
        return (rows["y1"].astype(float) - rows["y0"].astype(float)).to_numpy() > 0
    return rows["ite_proxy"].astype(float).to_numpy() > 0


def paper_card_indices(states: np.ndarray, margin: np.ndarray, rows: pd.DataFrame | None = None) -> dict[str, int]:
    """The decision points illustrated in the paper (Section 6, Figs. 3-4 and
    the two-level cards), given the policy's margin Delta Q on each pool
    state: the *median* decision to act (the intervene state whose Delta Q
    is the median of its side) and the median decision to wait. When
    ``rows`` is given, only decisions that agree with the oracle
    (``oracle_acts``) are eligible, so the illustrated act is one the
    intervention would have helped and the illustrated wait is one it would
    not. Stable sorts, so every script (figures/make_figures.py,
    run_xai_suite.py, run_risk_suite.py) shows the same cases regardless of
    numpy version. A side with no eligible state is omitted."""
    acts = margin > 0
    ok = np.ones(len(margin), bool) if rows is None else (acts == oracle_acts(rows))
    picks = {}
    for tag, idx in (("act", np.flatnonzero(acts & ok)), ("wait", np.flatnonzero(~acts & ok))):
        if len(idx) == 0:
            continue
        order = idx[np.argsort(np.abs(margin[idx]), kind="stable")]
        picks[tag] = int(order[len(order) // 2])
    return picks


def cycled_resources(n_rows: int, n_resources: int = N_RESOURCES) -> np.ndarray:
    """0, 1, ..., n_resources, 0, 1, ... across the rows (mean = n_resources / 2)."""
    return (np.arange(n_rows) % (n_resources + 1)).astype(np.float64)


def bpic_states(rows: pd.DataFrame, resources: np.ndarray, extra: list[str] | tuple[str, ...] = ()) -> np.ndarray:
    rel = (rows["prefix_nr"].astype(float) / rows["case_length"].astype(float).clip(lower=1)).clip(0, 1)
    cols = [rel.to_numpy(), rows["reliability"].astype(float).to_numpy(),
            rows["deviation"].astype(float).to_numpy(), resources]
    cols += [rows[c].astype(float).to_numpy() for c in extra]
    return np.stack(cols, axis=1).astype(np.float32)


def bpic_pool(csv_path, n_cases: int = N_CASES, seed: int = POOL_SEED, variant: str = DEFAULT_VARIANT) -> tuple[np.ndarray, pd.DataFrame]:
    """Evaluation pool: one prefix per case, resources cycled 0..N_RESOURCES."""
    rows = sample_one_prefix_per_case(load_bpic(csv_path), "case_id", n_cases, seed)
    return bpic_states(rows, cycled_resources(len(rows)), VARIANTS[variant]), rows


def bpic_full(csv_path, treatment_col: str, sample: int | None = None, seed: int = 42, variant: str = DEFAULT_VARIANT):
    """Full pool: every decision point, resources held at the training value.

    Returns (states, ite, historical_action). ``ite = y1 - y0`` is the row's
    own counterfactual outcome pair from the log's causal-effect estimate.
    """
    df = load_bpic(csv_path)
    states = bpic_states(df, np.full(len(df), float(N_RESOURCES)), VARIANTS[variant])
    ite = (df["y1"].astype(float) - df["y0"].astype(float)).to_numpy()
    hist = df[treatment_col].astype(int).to_numpy()
    if sample is not None and len(states) > sample:
        idx = np.random.default_rng(seed).choice(len(states), size=sample, replace=False)
        states, ite, hist = states[idx], ite[idx], hist[idx]
    return states, ite, hist


# ---------------------------------------------------------------------------
# SimBank Time-contact-HQ (resource-augmented log)
# ---------------------------------------------------------------------------


def load_simbank(pkl_path=None) -> pd.DataFrame:
    """The augmented log with the derived columns the checkpoint was trained on
    (prefix_nr, case_length, reliability, deviation, ite_proxy), built by the
    environment class itself so the two can never drift apart."""
    return SimBankHQEnvFast(pkl_path or paths.SIMBANK_PKL)._df


def simbank_decision_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["activity"].isin(DECISION_ACTIVITIES)]


def simbank_states(rows: pd.DataFrame, extra: list[str] | tuple[str, ...] = ()) -> np.ndarray:
    rel = (rows["prefix_nr"].astype(float) / rows["case_length"].astype(float).clip(lower=1)).clip(0, 1)
    cols = [rel.to_numpy(), rows["reliability"].astype(float).to_numpy(),
            rows["deviation"].astype(float).to_numpy(),
            rows["available_resources"].astype(float).to_numpy()]
    cols += [rows[c].astype(float).to_numpy() for c in extra]
    return np.stack(cols, axis=1).astype(np.float32)


def _simbank_rows(df: pd.DataFrame, rows: str) -> pd.DataFrame:
    if rows == "all":
        return df
    if rows == "decision":
        return simbank_decision_rows(df)
    raise ValueError(rows)


def simbank_pool(pkl_path=None, n_cases: int = N_CASES, seed: int = POOL_SEED, rows: str = "all",
                 variant: str = DEFAULT_VARIANT) -> tuple[np.ndarray, pd.DataFrame]:
    """Evaluation pool: one event per case, ``n_cases`` cases."""
    picked = sample_one_prefix_per_case(_simbank_rows(load_simbank(pkl_path), rows), "case_nr", n_cases, seed)
    return simbank_states(picked, VARIANTS[variant]), picked


def simbank_full(pkl_path=None, rows: str = "all", sample: int | None = None, seed: int = 42, variant: str = DEFAULT_VARIANT):
    """Full pool. ``rows="all"``: every event, the convention the checkpoint
    was trained under; ``rows="decision"``: the log's recorded contact
    decisions (diagnostic only, see module docstring).
    Returns (states, ite, historical_action, has_resources); ``ite`` is
    ``y1 - y0`` from the estimator's columns when the log has them (effect
    pkl) and the uncertainty proxy otherwise."""
    df = _simbank_rows(load_simbank(pkl_path), rows)
    states = simbank_states(df, VARIANTS[variant])
    ite = ((df["y1"].astype(float) - df["y0"].astype(float)) if "y1" in df.columns else df["ite_proxy"].astype(float)).to_numpy()
    hist = (df["activity"] == SIMBANK_TREATMENT_ACTIVITY).astype(int).to_numpy()
    has_res = (df["available_resources"].astype(float) > 0).to_numpy()
    if sample is not None and len(states) > sample:
        idx = np.random.default_rng(seed).choice(len(states), size=sample, replace=False)
        states, ite, hist, has_res = states[idx], ite[idx], hist[idx], has_res[idx]
    return states, ite, hist, has_res
