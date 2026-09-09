"""Augment SimBank's resource-augmented log with the retrained effect
estimator's counterfactual probabilities, so the SimBank agent can read the
same two state features as the BPIC agents and the reward can use a fitted
effect instead of the uncertainty proxy.

For every event of every case the two-model estimator of effect_model.py
(fit on the prefixes both arms share, see effect_model.effect_rows) gives

    Proba_if_Treated   = p_T(x_t) = P(cancelled | HQ contacted, prefix)
    Proba_if_Untreated = p_U(x_t) = P(cancelled | HQ skipped,   prefix)  (= 1 in this log)

named as in Shoush & Dumas's RL files so pools.VARIANTS applies unchanged.
The reward's per-row pair follows their rule, y = 1[predicted good outcome],
which on SimBank (where the semantics of the probabilities are known: Y = 1
is the *cancelled* application) reads

    y1 = 1[p_T < 0.5],   y0 = 1[p_U < 0.5],   ite = y1 - y0,

so intervening pays exactly where the estimator expects the contact to turn
a likely cancellation into a likely acceptance. Rows after the case's
decision (or of the priority cases that never reach it) are scored too --
the estimator extrapolates there, as available_resources does on BPIC.

Usage: python simbank_resources/add_effect_features.py   -> paths.SIMBANK_EFFECT_PKL
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import effect_model as em  # noqa: E402
import paths  # noqa: E402
import risk_model as rm  # noqa: E402


def main(out_path: Path) -> None:
    df, conf = rm.load_events("SimBank")
    X, meta = rm.encode_prefixes(df, conf)
    arms, feats = em.load_model("SimBank")
    Xd = em.one_hot(X[feats["raw_columns"]], feats["cat_cols"], feats["columns"])
    pT = arms["treated"].predict_proba(Xd)[:, 1]
    pU = arms["untreated"].predict_proba(Xd)[:, 1]
    scored = pd.DataFrame({"case_nr": meta["case_id"].astype(int).to_numpy(), "prefix_nr": meta["prefix_nr"].to_numpy(),
                           "Proba_if_Treated": pT.astype(np.float32), "Proba_if_Untreated": pU.astype(np.float32)})
    scored["y1"] = (scored["Proba_if_Treated"] < 0.5).astype(int)
    scored["y0"] = (scored["Proba_if_Untreated"] < 0.5).astype(int)

    raw = pd.read_pickle(paths.SIMBANK_PKL)
    raw = raw.sort_values(["case_nr", "synthetic_time_days"]).reset_index(drop=True)
    raw["prefix_nr"] = raw.groupby("case_nr").cumcount() + 1  # exactly SimBankHQEnvFast's numbering
    out = raw.merge(scored, on=["case_nr", "prefix_nr"], how="left", validate="one_to_one")
    assert out["Proba_if_Treated"].notna().all(), "every event must be scored"
    out = out.drop(columns=["prefix_nr"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(out_path)
    fit = em.effect_rows("SimBank", meta)
    print(f"scored {len(out)} events of {out.case_nr.nunique()} cases -> {out_path}")
    print(f"  p_T mean {pT.mean():.3f} (fit rows {pT[fit].mean():.3f}), p_U mean {pU.mean():.3f}; "
          f"ite>0 share all rows {(scored.y1 - scored.y0 > 0).mean():.3f}, fit rows {((scored.y1 - scored.y0) > 0)[fit].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(paths.SIMBANK_EFFECT_PKL))
    a = ap.parse_args()
    main(Path(a.out))
