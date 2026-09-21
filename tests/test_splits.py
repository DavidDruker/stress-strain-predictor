"""Leakage tests. These protect the only number the project actually reports."""

from __future__ import annotations

import numpy as np
import pytest

from stresspredict import features, splits


@pytest.mark.parametrize("protocol_name", ["gkf_grade", "lofo_family", "loso_source"])
def test_grouped_protocols_never_share_a_group(synthetic_grouped, protocol_name):
    """No group may appear in both sides of any fold.

    This is the test that makes the headline claim defensible. If a grade can be
    in training and in test at once then, under a composition-only feature set,
    the model is being scored on rows whose exact feature vector it memorised.
    """
    df, comp_long = synthetic_grouped
    protocol = splits.get(protocol_name)
    X = features.design_frame(df, comp_long)
    groups = protocol.groups(df)
    splitter = protocol.splitter()

    n_folds = 0
    for train_idx, test_idx in splitter.split(X, None, groups):
        n_folds += 1
        assert not set(groups[train_idx]) & set(groups[test_idx])
        assert not set(train_idx) & set(test_idx)
    assert n_folds >= 2


def test_random_kfold_is_declared_leaky():
    """The leaky protocol must say so in code, not only in prose."""
    assert splits.get("random_kfold").leaky is True
    assert all(not splits.get(n).leaky for n in ("gkf_grade", "lofo_family", "loso_source"))
    assert splits.HEADLINE_PROTOCOL == "gkf_grade"


def test_min_size_lofo_excludes_undersized_families(synthetic_grouped):
    """Families too small to score are kept in training, never held out alone."""
    df, _ = synthetic_grouped
    groups = df["family_group"].to_numpy(dtype=object)
    splitter = splits.MinSizeLeaveOneGroupOut(min_test_rows=10)

    excluded = splitter.excluded_groups(groups)
    assert "exotic" in excluded, "the 3-row synthetic family should be excluded"

    held_out = set()
    for train_idx, test_idx in splitter.split(np.zeros((len(df), 1)), None, groups):
        assert len(test_idx) >= 10
        held_out.update(groups[test_idx])
        # the undersized family must still be available to learn from
        assert "exotic" in set(groups[train_idx])
    assert "exotic" not in held_out


def test_inner_cv_is_grouped():
    """Tuning must be grouped, or the outer grouping is cosmetic."""
    from sklearn.model_selection import GroupKFold
    assert isinstance(splits.inner_cv(3), GroupKFold)


def test_calibration_holdout_is_grouped_and_sized(synthetic_grouped):
    """Phase 3's calibration split must not share grades with the rest."""
    df, _ = synthetic_grouped
    mask = splits.reserve_calibration_holdout(df, frac=0.2)
    held = set(df.loc[mask, "grade_group"])
    rest = set(df.loc[~mask, "grade_group"])
    assert not held & rest, "a grade appears in both calibration and training"
    assert 0.10 <= mask.mean() <= 0.32


def test_calibration_holdout_is_deterministic(synthetic_grouped):
    df, _ = synthetic_grouped
    a = splits.reserve_calibration_holdout(df, frac=0.15, seed=1)
    b = splits.reserve_calibration_holdout(df, frac=0.15, seed=1)
    c = splits.reserve_calibration_holdout(df, frac=0.15, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
