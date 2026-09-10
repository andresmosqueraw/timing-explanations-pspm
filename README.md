# timing-explanations-pspm

Supporting code for the paper *"How Risk, Effect and Timing Explanations
Compose in Prescriptive Process Monitoring Policies"* (RETIME).

The paper reads three explanations on the same decision point of a
resource-constrained PPO policy (Shoush & Dumas's design): the *timing
justification* (`φ^ΔQ`, `φ^ΔQ_wait`, Integrated Gradients on the margin for
acting over waiting), the *risk explanation* of the outcome predictor whose
outputs the policy reads as state, and the *effect explanation* of the
causal-effect estimator whose outputs it also reads; and asks how they
compose. `compose.py` propagates the timing justification to the prefix
attributes through the two lower levels (DeepSHAP's rescale rule, as Chen
et al. propagate Shapley values through a series of models), checks the
result against a direct attribution of the whole prefix-to-margin chain with
a deletion test, and cross-tabulates the decisions by (at risk) × (treatable)
× (acts).

## The explained policy

The paper's policy (`pools.DEFAULT_VARIANT = "cate_retrained"`) observes the
released four features plus the causal estimate's two counterfactual outcome
probabilities (`Proba_if_Treated`, `Proba_if_Untreated`) and is trained one
case per episode with `--reward-scale 0.01 --ent-coef 0.3` (600k steps on
BPIC2017, where 300k over-intervenes; see `models/variants/*_manifest.json`)
(`models/variants/ppo_<log>_cate_retrained.zip`) on the *coherent* state:
`build_retrained_state.py` scores the prepared log's temporal test split with
the retrained outcome predictor (`risk_model.py`) and effect estimator
(`effect_model.py`) into an RL CSV of Shoush & Dumas's format
(`data/retrained_state_<log>.csv`), so the risk and effect levels explain
exactly the numbers the policy reads. Every script defaults to it.
`--variant cate` is the same recipe on the shipped CSVs (`*_cate.json`):
there the retrained lower levels explain *other* numbers than the state
carries (corr 0.02 on r and 0.07 on p_T on BPIC2017), and rebuilding the
state from them flips 26 % / 39 % of the pool's decisions on BPIC2012 /
BPIC2017 (`compose_results_cate.json`), which is why the coherent variant is
the paper's. `--variant released` evaluates the earlier four-feature design
(`*_released.json`). SimBank keeps its single checkpoint, trained on the
retrained estimator's features (`simbank_resources/add_effect_features.py`).

Per decision point (`gain_table_results.json`) the paper's policy earns
42.3 / 39.9 / 46.3 (BPIC2012 / BPIC2017 / SimBank) against 24.5 / 24.8 /
-63.3 for always waiting, -49.5 / -49.8 / 16.6 for always intervening and
-27.6 / -32.0 / -61.4 for the recorded action (oracle 58.3 / 54.4 / 46.5). It
intervenes on 11.3 % / 15.9 % / 73.2 % of the decision points (precision
0.94 / 0.73 / 1.00, recall 0.55 / 0.69 / 1.00 against the positive-effect
rows). Both BPIC checkpoints are trained for 600k steps: at 300k the
BPIC2017 agent over-intervened (33.6 %, precision 0.38, gain 21.4), reading
`Proba_if_Treated` far more than `Proba_if_Untreated`, and lower entropy
coefficients collapse to never-intervene (`models/variants/*_manifest.json`,
`*_300k` checkpoints archived).

## Layout

- `paths.py` — every input and output path, overridable by environment
  variable (`TIMING_BPIC2012_CSV`, `TIMING_BPIC2017_CSV`,
  `TIMING_SIMBANK_RAW`, `TIMING_SIMBANK_PKL`, `TIMING_PAPER_FIGURES`).
- `pools.py` — the state pools every script evaluates on, defined once:
  the 500-case evaluation pool (Sections 6 and 7) and the full
  decision-point pools (Section 5.3). Its docstring states the two
  `available_resources` conventions and why SimBank is evaluated at every
  event.
- `dual_level.py`, `test_dual_level.py` — the differentiable heads
  (`MarginHead`, `WaitMarginHead`, ...), Integrated Gradients, and the
  `deletion_test` fidelity protocol.
- `foreign/train_ppo_fast_rl_prescriptive_monitoring.py` — the
  environment (Shoush & Dumas's state and reward, reimplemented from their
  CSV) and training script that produced the BPIC2012 and BPIC2017
  checkpoints.
- `models/` — the retrained BPIC2012 and BPIC2017 checkpoints, each with
  a `_manifest.json` (seed, step count, provenance), `_monitor.csv`, and
  `_training_curve.csv`.
- `simbank_resources/` — the SimBank Time-contact-HQ checkpoint and its
  own resource-augmentation script (`build_resources.py`, a synthetic
  capacity extension not part of SimBank's original design; see its
  docstring), training script (`train_ppo_simbank.py`), and a small
  reward-parameter diagnostic (`tune_cost.py`).
- `fidelity_test.py` — the deletion test (guided / random / anti-guided
  masking) on `φ^ΔQ_wait` across all three checkpoints, plus the
  per-feature share of mean |φ| (Table "card"); writes
  `fidelity_results.json`.
- `compute_gain_table.py` — mean reward per decision point under the
  historically recorded action versus the retrained policy, across all
  three logs; writes `gain_table_results.json`.
- `figures/make_figures.py` — the paper's figures (MDP timeline diagram,
  training-curve plot, two explanation cards).
- `run_xai_suite.py`, `xai_methods.py`, `xai_metrics.py`, `xai_plots.py`,
  `test_xai_suite.py` — the explanation methods and explanation-quality
  metrics of the PPM-explainability literature, applied to the same
  checkpoints and the same evaluation pool (see below); writes
  `xai_suite_results.json` and `figures/out/xai/<log>/`.
- `risk_model.py`, `run_risk_suite.py`, `test_risk_model.py` — the risk
  level: the CatBoost outcome predictor behind the policy's
  `reliability`/`deviation` features, retrained per log with Shoush &
  Dumas's own split, encoding and settings (`foreign/common_files/` is
  their code), explained with TreeSHAP and evaluated with the same
  literature metrics plus the deletion test; two-level cards (risk +
  timing) for chosen decision points. Writes `models/risk/`,
  `risk_results.json`, `figures/out/risk/<log>/`.
- `effect_model.py`, `run_effect_suite.py` — the effect level: the
  CausalLift two-model estimator (XGBoost per treatment arm with IPW from
  a logistic propensity) behind the policy's `Proba_if_Treated` /
  `Proba_if_Untreated` features, retrained per BPIC log with Shoush &
  Dumas's split and encoding, explained with TreeSHAP per arm and
  attribute-then-subtract for the effect, evaluated with the deletion test;
  three-level cards (risk | effect | timing). Writes `models/effect/`,
  `effect_results.json`, `figures/out/effect/<log>/`. On SimBank the
  estimator is fit on the four events both arms share (the log's recording
  policy contacts HQ always at event 5 and skips at event 9/10) and the
  untreated arm is constant (skipping always cancels);
  `simbank_resources/add_effect_features.py` scores every event into the
  effect-augmented pkl the SimBank `cate` policy is trained and evaluated on.
- `compose.py`, `run_compose_suite.py`, `test_compose.py` — the composition
  study (paper Section 6): the state rebuilt from the retrained models, the
  propagation of `φ^ΔQ` to the prefix attributes through `φ^r`, `φ^{p_T}`,
  `φ^{p_U}` with the channel of every contribution, the end-to-end function
  F(x, n) = ΔQ(s(x, n)) with a Shapley-sampling direct attribution and the
  deletion test on it, the (risky × treatable × acts) typology and the
  risk-vs-effect agreement on the shared prefix vocabulary. Writes
  `compose_results.json`, `figures/out/compose/<log>/`.
  The direct attribution of F is Shapley sampling against the pool as
  background (the same reference the levels use); `compose.anchor_to_direct`
  is the per-decision safeguard (direct ranking, propagated channel split),
  `compose.cancellation` the per-attribute cancellation index,
  `compose.well_defined` / `baseline_alignment` the two conditions of the
  operator, `sign_groups` the agree / oppose / independent split of the
  risk-vs-effect signs. `--variant risk_retrained` runs it on the four-feature
  policy (same recipe, no effect features), the composition with a live risk
  channel (`compose_results_risk_retrained.json`).
- `build_retrained_state.py` — the coherent-pipeline RL CSVs
  (`data/retrained_state_<log>.csv`) the `cate_retrained` policies are
  trained and evaluated on.
- `figures/make_composition_flow.py` — the paper's flow figure: the propagated
  card of one decision, prefix attributes → state coordinates by level →
  margin.
- `models/variants/` — policy checkpoints that depart from the released
  design (`pools.VARIANTS`). Both append the causal model's counterfactual
  outcome probabilities to the state; `cate` trains with one case per
  episode (`--episode case`), `cate_stream` with the released stream
  episodes (kept as the negative result: it never intervenes either, see
  below). `fidelity_test.py --variant <v>` and `compute_gain_table.py
  --variant <v>` evaluate them.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest test_dual_level.py -v
python fidelity_test.py            # deletion test + feature shares, all three logs
python compute_gain_table.py       # historical-vs-policy gain recomputation
TIMING_PAPER_FIGURES=<paper>/figures python figures/make_figures.py
pytest test_xai_suite.py test_risk_model.py -v
python run_xai_suite.py            # literature XAI methods + metrics, all three logs (~10 min)
python risk_model.py               # retrain the outcome (risk) predictors -> models/risk/ (BPIC2017 takes ~1 h)
python run_risk_suite.py           # risk explanations + metrics + two-level cards -> risk_results.json
python effect_model.py             # retrain the causal-effect estimators -> models/effect/ (BPIC2017 takes ~75 min)
python run_effect_suite.py         # effect explanations + deletion test + three-level cards -> effect_results.json
python build_retrained_state.py    # coherent-pipeline RL CSVs -> data/retrained_state_<log>.csv (needs the prepared logs)
python run_compose_suite.py        # composition study -> compose_results.json, figures/out/compose/
TIMING_PAPER_FIGURES=<paper>/figures python figures/make_three_level.py        # the paper's three-level card (fig5)
TIMING_PAPER_FIGURES=<paper>/figures python figures/make_composition_flow.py   # the paper's composition flow figure (fig6)
TIMING_PAPER_FIGURES=<paper>/figures python figures/make_mockup.py             # web mock-up figure (not in the paper)
```

Both paper figures read `risk_results.json` and `effect_results.json` and take their
plain-language feature readings from `figures/plain.py`; `run_effect_suite.py` also
writes a technical three-level card per log under `figures/out/effect/<log>/`.

Effect estimator (retrained, `models/effect/*_manifest.json`): per-arm held-out AUC
0.72 / 0.68 on BPIC2012 and 0.97 / 0.87 on BPIC2017; the retrained `p_T`, `p_U`
correlate 0.65 / 0.67 with the shipped columns on BPIC2012 (the RL CSV is the test
split) and not at all on BPIC2017 (the shipped file comes from a different run), but
the reward's positive-effect rule agrees on 77 % / 80 % of the test rows. Deletion test
on the effect log-odds ratio: gap 29 / 23 SE (BPIC2012, k = 1 / 3) and 30 / 31 SE
(BPIC2017), anti-guided 0; top attributes CreditScore_max 27 %, MonthlyCost_max 23 %,
ApplicationType 19 % on BPIC2017; open_cases_max 16 %, AMOUNT_REQ 9 % on BPIC2012.
SimBank (fit on events 1-4 of the 93,337 cases that reach the contact/skip decision,
186,629 training prefixes, 37 % treated): treated-arm AUC 0.95, untreated arm constant
(p_U = 1: every skipped case is cancelled); deletion gap 27 / 29 SE (k = 1 / 3), the
good-outcome verdict 1[p_T < 0.5] flipped on 42 % / 71 % of the prefixes; top attributes
amount 65 %, est_quality_max 17 %. Timing fidelity on the SimBank `cate` policy: act side
372 pool states (gap 43 SE at k = 1), wait side 128 (63 SE, every decision reversed);
p_T carries 58 % / 80 % of the attribution, reliability 22 % / 12 %.

## Findings about the released data that the code depends on

- The RL CSVs' `reliability` and `deviation` are outputs of the released
  offline phase's CatBoost outcome predictor (ensemble size 1):
  `deviation = 1 - predicted`, with class 1 = the *undesired* outcome
  (application not accepted), so `deviation = 1` means "predicted to end
  well". On BPIC2012 `reliability` is identical to `deviation` (ensemble
  agreement of one model); on BPIC2017 it is the probability of the
  predicted class, and `deviation = 1` on 99.2 % of rows.
- The released predictive-model configuration feeds CatBoost the column
  `time_to_event_m`, the time remaining until the case's treatment event
  (future information). `risk_model.py` drops it (`LEAK_NOTE`): with it the
  BPIC2017 predictor reaches AUC 0.94 with 71 % of its importance on that
  one column.
- The shipped BPIC2017 RL CSV does not come from the same predictor run as
  the prepared log: only 586k of its 1.2M rows match a (case, prefix) of the
  prepared log and 22 % of those disagree on the label, so its
  `predicted_proba_1` cannot be reproduced; BPIC2012's CSV is exactly the
  prepared log's test split (labels match on all 28,692 rows).
- Under the released 4-feature state and reward, always-wait is the
  optimal policy: intervening pays only when `y1 - y0 > 0` (19 % of
  BPIC2012 rows, 3 % of BPIC2017 rows), the reward's break-even is at
  P(positive effect) ≈ 0.6, and no observable cell of the state exceeds
  0.27. Whether `y1 - y0 > 0` is a deterministic function of
  `Proba_if_Treated`/`Proba_if_Untreated`, which the released state omits;
  the `cate` variants add them.
- Adding that information is not enough under the released *episode*
  structure: an episode streams the timestamp-sorted events of all cases and
  ends when any case finishes or the agent intervenes, and not intervening
  pays +50/+100 on every row whose effect is <= 0, so intervening forfeits
  the positive rewards of unrelated cases and `cate_stream` never
  intervenes (P(intervene) = 0.0000 on every row). `--episode case` makes
  the episode one case, the MDP the paper states; a per-case oracle then
  beats always-wait in 35 % of BPIC2012 cases (mean return 927 vs 594) and
  3.6 % of BPIC2017 cases (2086 vs 1969).
- Even then SB3's default `ent_coef=0` collapses the actor to
  never-intervene within the first few thousand steps (`cate_noent`):
  random interventions are punished on the 81-97 % of rows whose effect is
  <= 0 before the state can tell those rows apart, and the actor never
  explores again, although its critic learns that positive-effect states
  are worth less (V = 28 vs 227 on BPIC2012). `ent_coef=0.02` on the
  unscaled ±100 rewards collapses the same way (`cate_ent002`). What works
  (`cate`) is `--reward-scale 0.01 --ent-coef 0.3`: entropy stays near 0.2
  nats and the deterministic policy intervenes on 20.1 % of BPIC2012's
  decision points (precision 0.61, recall 0.63 against the positive-effect
  rows; gain 35.6 vs 24.3 always-wait, oracle 58.1) and 2.5 % of
  BPIC2017's (precision 0.98, recall 0.79; 55.9 vs 51.7, oracle 57.1).
  `fidelity_results_cate.json` holds the wait- and intervene-side deletion
  tests; the treatment-effect features carry 55-63 % of |φ| on BPIC2012.
  Evaluation always uses the unscaled reward and state.

## Literature explanation methods and metrics (`run_xai_suite.py`)

The suite reproduces, on this policy's timing decision, what five
PPM-explainability code bases produce for outcome predictors:

| Reference | Methods reproduced | Metrics reproduced |
|---|---|---|
| Warmuth & Leopold 2022 (text-based XPPM) | SHAP bar / beeswarm / waterfall / force | parsimony, monotonicity (Spearman), rediscovery rate (identity matching) |
| Elkhawaga et al. 2022 (PPM_XAI_Comparison) | SHAP global + local (force, decision, dependence), LIME + CSI/VSI stability, permutation importance (10 repeats, box plots), ALE, correlation heat map, mutual information, KMeans-selected local instances, timings | cross-method agreement (Spearman / Kendall / top-k Jaccard) |
| Stevens et al. (Quantifying-Explainability) | SHAP summary | parsimony, functional complexity (flip rate), monotonicity (Spearman and Kendall) |
| Elkhawaga et al. 2023 (ConsisXAI) | SHAP, permutation | reducts / core from nine scoring criteria, reduct- and core-consistency ratio (score-weighted and count-based), AIC / BIC |
| Stevens & De Smedt (X-MOP guidelines) | SHAP, permutation effects with observed-value replacement | parsimony per attribute family, functional complexity (L1) per family, monotonicity, LOD |

Every method explains the same target, by default the paper's wait-side
margin `φ^ΔQ_wait` on the wait states of the Section 6/7 pool (`--target`
switches to the intervene margin, `p_wait`, `p_intervene` or the critic).
The paper's Integrated Gradients attribution is scored with the same
metrics, and every method's ranking is put through the paper's deletion
test (`k=1`), so both evaluation traditions appear side by side in
`xai_suite_results.json`. The state's three attribute families
(`progress` = relative_position, `prediction` = reliability and deviation,
`capacity` = available_resources) stand in for the event / case /
control-flow families of the reference metrics. Two deliberate deviations
from the reference code are documented in the docstrings: LIME uses the
continuous surrogate by default (`--lime-discretize` restores LIME's
default, which fits some states here with R² below 0.01), and the
weight-of-evidence criterion uses the usual +0.5 count adjustment so a
perfectly separating feature is not dropped.

## Data

The scripts read three inputs that are too large for this repository:

| Input | Default location | Source |
|---|---|---|
| `ready_to_use_adaptive_bpic2012.csv` | `data/` | Zenodo archive (see paper's Data Availability) |
| `ready_to_use_adaptive_bpic2017.csv` | `data/` | Shoush & Dumas's repository, `rl/data/` (https://github.com/mshoush/RL-prescriptive-monitoring) |
| `simbank_time_contact_hq_with_resources.pkl` | `simbank_resources/data/` | Zenodo archive, or regenerate with `simbank_resources/build_resources.py --n-servers 5` from SimBank's as-generated `loan_log_[_time_contact_HQ_]_100000_train_normal` (placed in `data/` or pointed to with `TIMING_SIMBANK_RAW`) |

Set the matching `TIMING_*` variable from `paths.py` to use a file kept
elsewhere.

## Retraining the checkpoints

```bash
python foreign/train_ppo_fast_rl_prescriptive_monitoring.py                     # BPIC2017, 300k steps, seed 42
python foreign/train_ppo_fast_rl_prescriptive_monitoring.py \
    --csv data/ready_to_use_adaptive_bpic2012.csv \
    --out models/ppo_bpic2012_rl_prescriptive_monitoring                         # BPIC2012
python simbank_resources/train_ppo_simbank.py                                    # SimBank, 150k steps, seed 42
```

## License

This repository (code, trained checkpoints, manifests, and result JSONs)
is released under CC0 1.0 (public domain dedication); see `LICENSE`.
