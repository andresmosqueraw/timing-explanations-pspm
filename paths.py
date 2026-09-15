"""Every input this repository reads and every output it writes, in one place.

Defaults are relative to the repository root, so a fresh clone with the data
dropped into ``data/`` and ``simbank_resources/data/`` runs as is. Each path
can be overridden with the environment variable named next to it, for
checkouts that keep the third-party logs elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"


def _env_path(var: str, default: Path) -> Path:
    return Path(os.environ[var]).expanduser() if os.environ.get(var) else default


# --- inputs: third-party logs (not committed; see README "Data") ------------
# Shoush & Dumas's preprocessed BPIC logs (their repo's own CSVs, ';'-separated).
BPIC2012_CSV = _env_path("TIMING_BPIC2012_CSV", DATA / "ready_to_use_adaptive_bpic2012.csv")
BPIC2017_CSV = _env_path("TIMING_BPIC2017_CSV", DATA / "ready_to_use_adaptive_bpic2017.csv")
# The coherent-pipeline RL CSVs (build_retrained_state.py): the prepared
# log's test split scored by the retrained risk and effect models, the state
# the "cate_retrained" policy variant is trained and evaluated on.
RETRAINED_CSV = {"BPIC2012": _env_path("TIMING_BPIC2012_RETRAINED_CSV", DATA / "retrained_state_bpic2012.csv"),
                 "BPIC2017": _env_path("TIMING_BPIC2017_RETRAINED_CSV", DATA / "retrained_state_bpic2017.csv")}


# The validation split scored the same way: cases the agent never trained on,
# used only to report the policy's gain out of sample (compute_gain_table.py).
RETRAINED_VAL_CSV = {"BPIC2012": _env_path("TIMING_BPIC2012_RETRAINED_VAL_CSV", DATA / "retrained_state_bpic2012_val.csv"),
                     "BPIC2017": _env_path("TIMING_BPIC2017_RETRAINED_VAL_CSV", DATA / "retrained_state_bpic2017_val.csv"),
                     "Sepsis": _env_path("TIMING_SEPSIS_STATE_VAL_CSV", DATA / "retrained_state_sepsis_val.csv")}
# The training split scored out of fold (crossfit.py): the only state the
# agents train on. Its val / test counterparts are never used for training.
RETRAINED_TRAIN_CSV = {"BPIC2012": _env_path("TIMING_BPIC2012_RETRAINED_TRAIN_CSV", DATA / "retrained_state_bpic2012_train.csv"),
                       "BPIC2017": _env_path("TIMING_BPIC2017_RETRAINED_TRAIN_CSV", DATA / "retrained_state_bpic2017_train.csv"),
                       "Sepsis": _env_path("TIMING_SEPSIS_STATE_TRAIN_CSV", DATA / "retrained_state_sepsis_train.csv")}
# Cross-fitting record (folds, fold sizes, settings), committed.
CROSSFIT_JSON = REPO / "crossfit_manifest.json"
# The agents' hyper-parameter selection on the validation split, committed.
POLICY_SELECTION_JSON = REPO / "policy_selection.json"


def retrained_csv(log: str, split: str = "test"):
    test = {**RETRAINED_CSV, "Sepsis": SEPSIS_STATE_CSV}
    return {"train": RETRAINED_TRAIN_CSV, "val": RETRAINED_VAL_CSV, "test": test}[split][log]


def bpic_csv(log: str, variant: str = "cate_retrained"):
    """The RL CSV a BPIC policy variant reads: the coherent-pipeline CSV for
    ``cate_retrained``, Shoush & Dumas's shipped CSV for every other variant."""
    if variant in ("cate_retrained", "risk_retrained"):
        return retrained_csv(log)
    return {"BPIC2012": BPIC2012_CSV, "BPIC2017": BPIC2017_CSV}[log]


# SimBank's as-generated Time-contact-HQ log (De Moor et al.), input of
# simbank_resources/build_resources.py ...
SIMBANK_RAW = _env_path("TIMING_SIMBANK_RAW", DATA / "loan_log_[_time_contact_HQ_]_100000_train_normal")
# ... and that script's output, the resource-augmented log every other
# SimBank step reads.
SIMBANK_PKL = _env_path(
    "TIMING_SIMBANK_PKL", REPO / "simbank_resources/data/simbank_time_contact_hq_with_resources.pkl"
)

# Sepsis Cases - Event Log (Mannhardt, 4TU.ResearchData, doi:10.4121/uuid:915d2bfb-
# 7e84-49ad-a286-dc35f063a460) parsed to one row per event (columns case_id,
# activity, timestamp, org:group and the event attributes; see
# datasets_eda/README.md); git-ignored like the other logs.
SEPSIS_EVENTS_PARQUET = _env_path("TIMING_SEPSIS_PARQUET", DATA / "sepsis_events.parquet")
# The coherent-pipeline RL CSV for Sepsis (build_sepsis_state.py): the
# prepared log's temporal test split scored by the retrained risk and effect
# models, in the same format as RETRAINED_CSV above. There is no "shipped"
# counterpart (see build_sepsis_state.py docstring) -- this is the only state
# Sepsis's policy ever reads.
SEPSIS_STATE_CSV = _env_path("TIMING_SEPSIS_STATE_CSV", DATA / "retrained_state_sepsis.csv")

# Shoush & Dumas's *prepared* event logs (inputs of their predictive model;
# their owncloud archive, PeerJ2023/prepared_data/<log>/prepared_treatment_
# outcome_time_to_event_<log>.csv). Read by risk_model.py to retrain the
# outcome predictor whose output the policy's reliability/deviation come from.
PREPARED_LOGS = _env_path("TIMING_PREPARED_LOGS", REPO / "PeerJ2023/prepared_data")

# --- committed checkpoints --------------------------------------------------
MODELS = REPO / "models"
BPIC2012_MODEL = MODELS / "ppo_bpic2012_rl_prescriptive_monitoring.zip"
BPIC2017_MODEL = MODELS / "ppo_bpic2017_rl_prescriptive_monitoring.zip"
BPIC2017_MANIFEST = MODELS / "ppo_bpic2017_rl_prescriptive_monitoring_manifest.json"
BPIC2017_CURVE = MODELS / "ppo_bpic2017_rl_prescriptive_monitoring_training_curve.csv"
SIMBANK_MODEL = REPO / "simbank_resources/models/ppo_simbank_time_contact_hq.zip"
# SimBank log augmented with the retrained effect estimator's p_T / p_U (and
# the reward's y1 / y0 derived from them), built by
# simbank_resources/add_effect_features.py; read by every non-released variant
SIMBANK_EFFECT_PKL = _env_path("TIMING_SIMBANK_EFFECT_PKL", REPO / "simbank_resources/data/simbank_time_contact_hq_with_effect.pkl")


def simbank_pkl(variant: str = "cate"):
    return SIMBANK_PKL if variant == "released" else SIMBANK_EFFECT_PKL
# Policy variants that depart from the released design (pools.VARIANTS), e.g.
# models/variants/ppo_bpic2017_cate.zip
VARIANT_MODELS = MODELS / "variants"


def variant_model(log: str, variant: str = "cate") -> Path:
    """Checkpoint of ``variant`` for ``log``. ``released`` is the 4-feature
    design (SimBank: the uncertainty-proxy checkpoint); the other variants
    live under models/variants/ppo_<log>_<variant>.zip."""
    if variant == "released":
        return {"BPIC2012": BPIC2012_MODEL, "BPIC2017": BPIC2017_MODEL, "SimBank": SIMBANK_MODEL}[log]
    if log in ("SimBank", "Sepsis") and variant == "cate_retrained":
        # SimBank's cate checkpoint already reads the retrained estimator's
        # features (add_effect_features.py): its pipeline is coherent as is.
        # Sepsis is built coherent from scratch (build_sepsis_state.py) and
        # has a single checkpoint too -- no shipped/rebuilt distinction ever
        # existed for it, so "cate_retrained" and "cate" name the same file.
        variant = "cate"
    return VARIANT_MODELS / f"ppo_{log.lower()}_{variant}.zip"


def variant_artifact(log: str, variant: str, suffix: str) -> Path:
    """Sibling of the checkpoint: suffix in {"_manifest.json", "_training_curve.csv", "_monitor.csv"}."""
    return variant_model(log, variant).with_name(variant_model(log, variant).stem + suffix)


# Retrained outcome ("risk") predictors, one CatBoost per log (risk_model.py):
# models/risk/<log>_catboost.cbm + _manifest.json + _features.json
RISK_MODELS = MODELS / "risk"

# --- outputs ----------------------------------------------------------------
FIDELITY_JSON = REPO / "fidelity_results.json"
GAIN_JSON = REPO / "gain_table_results.json"
# Where figures/make_figures.py writes the PDFs. Point it at the paper's
# figures/ directory to regenerate the manuscript's figures in place.
PAPER_FIGURES = _env_path("TIMING_PAPER_FIGURES", REPO / "figures/out")
# run_xai_suite.py: the literature's explanation methods and metrics on the
# same checkpoints (SHAP, LIME, permutation, ALE + parsimony, functional
# complexity, monotonicity, LOD, ConsisXAI). One JSON, one figure folder per log.
XAI_JSON = REPO / "xai_suite_results.json"
XAI_FIGURES = _env_path("TIMING_XAI_FIGURES", REPO / "figures/out/xai")
# run_risk_suite.py: the risk (outcome) explanation and its evaluation.
RISK_JSON = REPO / "risk_results.json"
RISK_FIGURES = _env_path("TIMING_RISK_FIGURES", REPO / "figures/out/risk")
# run_effect_suite.py: the effect (causal-estimator) explanation, third level.
EFFECT_MODELS = MODELS / "effect"
EFFECT_JSON = REPO / "effect_results.json"
EFFECT_FIGURES = _env_path("TIMING_EFFECT_FIGURES", REPO / "figures/out/effect")
# run_compose_suite.py: how the three levels compose (propagated attribution,
# end-to-end fidelity, typology, shipped-vs-rebuilt state).
COMPOSE_JSON = REPO / "compose_results.json"
COMPOSE_FIGURES = _env_path("TIMING_COMPOSE_FIGURES", REPO / "figures/out/compose")
