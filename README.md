# timing-explanations-pspm

Supporting code for the paper *"Explaining When to Wait: A Timing
Framework for Explainability in Prescriptive Process Monitoring
Policies"* (Paper 1 — the narrow, Padella-style companion to the
evaluation-protocol paper).

This paper formalizes and illustrates **only** the timing/wait-vs-act
explanation (`φ^ΔQ`, `φ^ΔQ_wait`) for one already-published policy — the
resource-constrained agent of Shoush & Dumas — using Integrated
Gradients, justified by a Completeness-derived existence corollary.
It deliberately does **not** test explanation fidelity (no deletion
tests, no z-scores): that protocol is a separate paper's contribution.
See `faithful-pspm-explanations` (sibling repo) for that work — this
repo does not depend on it and should be kept independent.

## Layout

- `dual_level.py`, `test_dual_level.py`, `run_dual_level_ppo.py` — copied
  verbatim from `faithful-pspm-explanations/experiments/generality_ppo/`
  (same lineage, same tests). Not all of it is used by this paper (e.g.
  the multi-method/branch-wise comparison machinery is Paper 2's); kept
  whole rather than hand-trimmed, to avoid partial-copy bugs.
- `models/ppo_bpic2017_rl_prescriptive_monitoring.zip` — the retrained
  checkpoint this paper explains, with its `_manifest.json` (seed 42,
  300k steps), `_monitor.csv`, `_training_curve.csv` for provenance.
- `figures/make_figures.py` — generates the paper's 4 figures (MDP
  timeline diagram, training-curve plot, two explanation cards) and
  writes them directly into `../../../../3-escribir-paper/conferencias/paper1/figures/`
  (figures are a paper build artifact and live in the LaTeX tree; the
  generator lives here).

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest test_dual_level.py -v
python figures/make_figures.py   # regenerates the 4 paper figures
```

`figures/make_figures.py` reads the third-party BPIC2017 CSV from
`../libraries-prescriptive-process/01-rl-online/RL-prescriptive-monitoring/rl/data/ready_to_use_adaptive_bpic2017.csv`
(not committed here — Shoush & Dumas's own preprocessed data, sourced
from their repo, same as `generality_ppo`'s README documents).
