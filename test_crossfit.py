"""crossfit.case_folds: the out-of-fold split must keep every case whole."""

import numpy as np
import pandas as pd

import crossfit


def test_every_case_in_exactly_one_fold_and_folds_balanced():
    rng = np.random.default_rng(0)
    cases = pd.Series(rng.integers(0, 500, size=20000)).astype(str)
    fold = crossfit.case_folds(cases)
    per_case = pd.DataFrame({"case": cases, "fold": fold}).groupby("case")["fold"].nunique()
    assert per_case.max() == 1
    assert set(np.unique(fold)) == set(range(crossfit.N_FOLDS))
    n_cases = pd.DataFrame({"case": cases, "fold": fold}).drop_duplicates("case")["fold"].value_counts()
    assert n_cases.max() - n_cases.min() <= 1


def test_folds_are_reproducible():
    cases = pd.Series([f"c{i % 37}" for i in range(1000)])
    assert (crossfit.case_folds(cases) == crossfit.case_folds(cases)).all()
