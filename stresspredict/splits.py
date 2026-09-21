"""Evaluation protocols. This is the module that decides whether the headline
number means anything.

Under a composition-only feature set, two rows of the same grade at different
tempers are the SAME feature vector with different targets. A random K-fold puts
a row's exact twin in the training set, so the reported score measures grade
lookup. The gap between `random_kfold` and `gkf_grade` is not noise -- it is the
size of the leak, and it is reported deliberately rather than hidden.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

DEFAULT_SEED = 20260921


class MinSizeLeaveOneGroupOut:
    """Leave-one-group-out, skipping groups too small to score.

    SteelBench's 17 steel families include five with 1-3 rows. Scoring a fold of
    one row produces a number with no standard error that then contaminates any
    pooled mean. Undersized groups stay in the training set and are excluded from
    being held out; which ones is recorded, not silently swallowed.
    """

    def __init__(self, min_test_rows: int = 10):
        self.min_test_rows = min_test_rows

    def eligible_groups(self, groups: np.ndarray) -> list:
        vals, counts = np.unique(np.asarray(groups, dtype=object), return_counts=True)
        return [v for v, c in zip(vals, counts, strict=True) if c >= self.min_test_rows]

    def excluded_groups(self, groups: np.ndarray) -> dict:
        vals, counts = np.unique(np.asarray(groups, dtype=object), return_counts=True)
        return {str(v): int(c) for v, c in zip(vals, counts, strict=True) if c < self.min_test_rows}

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self.eligible_groups(groups))

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        groups = np.asarray(groups, dtype=object)
        for g in self.eligible_groups(groups):
            test = np.flatnonzero(groups == g)
            train = np.flatnonzero(groups != g)
            if len(train) and len(test):
                yield train, test


@dataclass(frozen=True)
class Protocol:
    name: str
    group_column: str | None
    question: str
    leaky: bool
    make_splitter: object

    def splitter(self):
        return self.make_splitter()

    def groups(self, df: pd.DataFrame) -> np.ndarray | None:
        if self.group_column is None:
            return None
        return df[self.group_column].to_numpy(dtype=object)


PROTOCOLS: dict[str, Protocol] = {
    "random_kfold": Protocol(
        name="random_kfold",
        group_column=None,
        question="Leaky upper bound ONLY. A row's same-grade twin can sit in training.",
        leaky=True,
        make_splitter=lambda: KFold(n_splits=5, shuffle=True, random_state=DEFAULT_SEED),
    ),
    "gkf_grade": Protocol(
        name="gkf_grade",
        group_column="grade_group",
        question="Can it predict a grade it has never seen? -- the headline protocol.",
        leaky=False,
        make_splitter=lambda: GroupKFold(n_splits=5),
    ),
    "lofo_family": Protocol(
        name="lofo_family",
        group_column="family_group",
        question="Does it transfer to an entirely unseen alloy family?",
        leaky=False,
        make_splitter=lambda: MinSizeLeaveOneGroupOut(min_test_rows=10),
    ),
    "loso_source": Protocol(
        name="loso_source",
        group_column="source_group",
        question="Does it survive a change of laboratory / publication bias?",
        leaky=False,
        make_splitter=lambda: LeaveOneGroupOut(),
    ),
}

HEADLINE_PROTOCOL = "gkf_grade"


def get(name: str) -> Protocol:
    if name not in PROTOCOLS:
        raise KeyError(f"unknown protocol {name!r}; known: {sorted(PROTOCOLS)}")
    return PROTOCOLS[name]


def inner_cv(n_splits: int = 3):
    """Splitter for hyper-parameter search INSIDE an outer training fold.

    Grouped, always. Tuning against a random inner split while reporting a
    grouped outer split is the single most common way a project like this leaks:
    the hyper-parameters are chosen using the very twins the outer split was
    built to separate.
    """
    return GroupKFold(n_splits=n_splits)


def reserve_calibration_holdout(
    df: pd.DataFrame,
    frac: float = 0.15,
    group_column: str = "grade_group",
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Reserve a grouped calibration holdout for Phase 3 conformal prediction.

    Defined in Phase 1 so Phase 3 needs no restructuring, and grouped by grade
    because a calibration set sharing grades with training gives a coverage
    guarantee that is arithmetically valid and practically fiction.

    Returns a boolean mask, True where the row is reserved for calibration.
    """
    rng = np.random.default_rng(seed)
    groups = df[group_column].to_numpy(dtype=object)
    unique = np.unique(groups)
    rng.shuffle(unique)

    target = int(round(frac * len(df)))
    chosen: set = set()
    n = 0
    for g in unique:
        if n >= target:
            break
        chosen.add(g)
        n += int((groups == g).sum())
    return np.isin(groups, list(chosen))
