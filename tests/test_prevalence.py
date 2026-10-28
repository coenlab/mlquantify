"""Tests for prevalence utilities (normalize_prevalence)."""
import warnings

import numpy as np
import pytest

from mlquantify.utils import normalize_prevalence


def test_normalize_prevalence_array():
    assert normalize_prevalence([2.0, 3.0, 5.0], classes=[0, 1, 2]) == pytest.approx(
        {0: 0.2, 1: 0.3, 2: 0.5})


def test_normalize_prevalence_dict():
    assert normalize_prevalence({0: 0.1, 1: 0.1, 2: 0.3}, classes=[0, 1, 2]) == pytest.approx(
        {0: 0.2, 1: 0.2, 2: 0.6})


def test_normalize_prevalence_emits_no_warning():
    """The old implementation raised numpy's 'where without out' UserWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        normalize_prevalence([0.4, 0.6], classes=[0, 1])
        normalize_prevalence(np.array([1.0, 2.0, 1.0]), classes=[0, 1, 2])
        normalize_prevalence({0: 1.0, 1: 1.0}, classes=[0, 1])


@pytest.mark.parametrize("prev", [[0.0, 0.0, 0.0], {0: 0.0, 1: 0.0, 2: 0.0}])
def test_normalize_prevalence_zero_sum_is_uniform(prev):
    """A non-positive sum must yield a valid (uniform) vector, not garbage."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = normalize_prevalence(prev, classes=[0, 1, 2])
    vals = np.array([result[c] for c in (0, 1, 2)], dtype=float)
    assert np.all(np.isfinite(vals))
    assert vals.sum() == pytest.approx(1.0)
    assert vals == pytest.approx([1 / 3, 1 / 3, 1 / 3])


@pytest.mark.parametrize("prev", [[0.0, 0.0, 0.0], np.zeros(3)])
def test_normalize_prevalences_zero_sum_is_uniform(prev):
    """The plural helper must honour the same contract as the singular one.

    ``normalize_prevalences`` is the one the quantifiers go through, via
    ``validate_prevalences``. It used to skip the division when the sum was
    zero and hand the all-zeros vector back as if it were an estimate.
    """
    from mlquantify.utils._validation import normalize_prevalences

    result = np.asarray(
        normalize_prevalences(np.asarray(prev, dtype=float), classes=[0, 1, 2]),
        dtype=float,
    )

    assert np.all(np.isfinite(result))
    assert result.sum() == pytest.approx(1.0)
    assert result == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_one_vs_rest_never_returns_a_zero_sum_vector():
    """End to end: no estimate may sum to zero, however hard the sample.

    In a small sample of a hard multiclass problem every binary quantifier of
    the one-vs-rest path can clip its own estimate to zero, and the combined
    vector then summed to zero. Measured on 0.5.1: 6 of 200 samples for ACC,
    61 for MS, and all 200 for CDE.
    """
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    from mlquantify._config import config_context
    from mlquantify.counting import ACC, MS
    from mlquantify.likelihood import CDE

    X, y = make_classification(
        n_samples=1200, n_features=8, n_informative=2, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, class_sep=0.2, flip_y=0.45,
        random_state=7,
    )
    X_train, y_train, X_test = X[:600], y[:600], X[600:]
    rng = np.random.default_rng(0)

    for Quantifier in (ACC, MS, CDE):
        q = Quantifier(LogisticRegression(max_iter=1000, random_state=42))
        q.fit(X_train, y_train)
        for _ in range(40):
            sample = X_test[rng.choice(len(X_test), 10, replace=False)]
            with config_context(prevalence_return_type="array"):
                estimate = np.asarray(q.predict(sample), dtype=float)
            assert np.all(np.isfinite(estimate)), Quantifier.__name__
            assert estimate.sum() == pytest.approx(1.0), Quantifier.__name__
