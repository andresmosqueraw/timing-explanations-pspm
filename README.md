# timing-explanations-pspm

Supporting code for the paper *"Explaining When to Wait: A Timing
Framework for Explainability in Prescriptive Process Monitoring
Policies"*.

The paper formalizes the timing/wait-vs-act explanation (`φ^ΔQ`,
`φ^ΔQ_wait`) for already-published resource-constrained PPO policies
(Shoush & Dumas's design), justified by a Completeness-derived existence
corollary (Integrated Gradients), and tests its fidelity with a deletion
test. It retrains and independently revalidates the policy's quality on
three logs: BPIC2012, BPIC2017, and SimBank's *Time contact HQ*
intervention.

## Layout

- `dual_level.py`, `test_dual_level.py`, `run_dual_level_ppo.py` — the
  differentiable heads (`MarginHead`, `WaitMarginHead`, ...), Integrated
  Gradients, and the `deletion_test` fidelity protocol used throughout.
- `models/` — the retrained BPIC2012 and BPIC2017 checkpoints, each with
  a `_manifest.json` (seed, step count, provenance), `_monitor.csv`, and
  `_training_curve.csv`.
- `simbank_resources/` — the SimBank Time-contact-HQ checkpoint and its
  own resource-augmentation script (`build_resources.py`, a synthetic
  capacity extension not part of SimBank's original design; see its
  docstring), training script (`train_ppo_simbank.py`), and a small
  reward-parameter diagnostic (`tune_cost.py`).
- `fidelity_test.py` — runs the deletion test (guided / random /
  anti-guided masking) on `φ^ΔQ_wait` across all three checkpoints;
  writes `fidelity_results.json`.
- `compute_gain_table.py` — recomputes mean reward per decision point
  under the historically recorded action versus the retrained policy,
  across all three logs; writes `gain_table_results.json`.
- `figures/make_figures.py` — generates the paper's figures (MDP timeline
  diagram, training-curve plot, two explanation cards) and writes them
  into the paper's LaTeX tree.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest test_dual_level.py -v
python fidelity_test.py            # deletion test on all three logs
python compute_gain_table.py       # historical-vs-policy gain recomputation
python figures/make_figures.py     # regenerates the paper's figures
```

Reproducing `fidelity_test.py`, `compute_gain_table.py`, and
`figures/make_figures.py` also requires the BPIC2012/BPIC2017 CSVs and
the SimBank resource-augmented log; these are too large for this
repository (see `.gitignore`) and are available from the authors on
request.

## License

This repository (code, trained checkpoints, manifests, and result JSONs)
is released under CC0 1.0 (public domain dedication); see `LICENSE`.
