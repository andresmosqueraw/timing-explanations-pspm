# Log screening

Copies of the dataset screening behind the paper's log selection (Section 3,
"Not every log fits this pipeline"), made on the parsed public logs of the
EDA project (`process-mining/datasets/eda/`, one parquet per log, not
redistributed here):

- `comparativa_T21.md` — first screening: non-saturated outcome, trace
  length, vocabulary, plausible intervention.
- `aptitud-datasets-2026-08-07.md` — the verdict per log with the added
  requirements (enough test decisions, a variable intervention position).
  Written for an earlier study of the same group, so it also scores criteria
  (off-policy evaluation) this paper does not use.

The Sepsis parquet the pipeline reads (`paths.SEPSIS_EVENTS_PARQUET`,
default `data/sepsis_events.parquet`) is that EDA project's parse of
"Sepsis Cases - Event Log" (Mannhardt, 4TU.ResearchData).
