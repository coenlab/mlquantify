"""Tests for the ``classes`` parameter of aggregative ``aggregate`` methods.

When ``aggregate`` is called with predictions that miss one or more classes,
passing ``classes=`` guarantees the output still reports every requested class
(absent classes get prevalence 0). The ``predict`` path is also checked across
every aggregative family to confirm the output always spans all fitted classes.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from mlquantify._config import config_context
from mlquantify.counting import (
    CC, PCC, ACC, FM, GACC, GPACC, MS, MS2, T50, TAC, TMAX, TX,
)
from mlquantify.likelihood import EMQ, CDE, MLPE
from mlquantify.neighbors import PWK
from mlquantify.matching import DyS, HDy, SMM, SORD


def _estimator():
    return LogisticRegression(max_iter=500)


def _make(Quantifier):
    """Construct a quantifier, supplying an estimator only when it takes one."""
    if Quantifier in (PWK, MLPE):  # no wrapped estimator: k-NN based, and
        return Quantifier()        # MLPE, which is non-aggregative
    return Quantifier(_estimator())


# Natively-multiclass quantifiers (counting, likelihood, neighbors). All
# aggregative except MLPE, which takes no estimator.
MULTICLASS_QUANTIFIERS = [
    CC, PCC, ACC, FM, GACC, GPACC, MS, MS2, T50, TAC, TMAX, TX,
    EMQ, CDE, MLPE, PWK,
]

# Binary-only quantifiers (matching family).
BINARY_QUANTIFIERS = [DyS, HDy, SMM, SORD]


def _assert_full_prevalence(prevalence, n_classes):
    prevalence = np.asarray(prevalence, dtype=float)
    assert prevalence.shape == (n_classes,)
    assert np.all(prevalence >= 0)
    assert prevalence.sum() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Explicit ``classes`` argument (self-contained aggregate calls)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("Quantifier", [CC, PCC])
def test_aggregate_classes_fills_absent_class(Quantifier):
    """A class absent from the predictions still appears (with prevalence 0)
    when requested via ``classes``."""
    X, y = make_classification(
        n_samples=600, n_classes=3, n_informative=6, n_redundant=0,
        n_clusters_per_class=1, random_state=1,
    )
    q = _make(Quantifier).fit(X[:400], y[:400])

    with config_context(prevalence_return_type="array"):
        if Quantifier is CC:
            # crisp labels covering only classes 0 and 1
            predictions = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        else:
            # posteriors with all the mass on classes 0 and 1
            predictions = np.tile([0.6, 0.4, 0.0], (8, 1))

        out = np.asarray(q.aggregate(predictions, classes=[0, 1, 2]), dtype=float)

    assert out.shape == (3,)
    assert out[2] == pytest.approx(0.0)
    assert out.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("Quantifier", [CC, PCC])
def test_aggregate_classes_none_uses_fitted(Quantifier):
    """With ``classes=None`` the fitted class set is used, so the output still
    spans all classes even if the predictions miss one."""
    X, y = make_classification(
        n_samples=600, n_classes=3, n_informative=6, n_redundant=0,
        n_clusters_per_class=1, random_state=2,
    )
    q = _make(Quantifier).fit(X[:400], y[:400])

    with config_context(prevalence_return_type="array"):
        predictions = (
            np.array([0, 0, 1, 1, 0])
            if Quantifier is CC
            else np.tile([0.7, 0.3, 0.0], (5, 1))
        )
        out = np.asarray(q.aggregate(predictions), dtype=float)

    _assert_full_prevalence(out, n_classes=3)


# MLPE is out: being non-aggregative it has no ``estimator_`` to read
# posteriors from, and its ``aggregate`` does not depend on them.
@pytest.mark.parametrize("Quantifier", [CC, PCC, EMQ])
def test_aggregate_classes_direct(Quantifier):
    """Quantifiers whose ``aggregate`` is self-contained honour ``classes`` and
    return a prevalence spanning exactly those classes."""
    X, y = make_classification(
        n_samples=600, n_classes=3, n_informative=6, n_redundant=0,
        n_clusters_per_class=1, random_state=3,
    )
    q = _make(Quantifier).fit(X[:400], y[:400])
    proba = q.estimator_.predict_proba(X[400:])

    # CC needs crisp labels; the rest accept posteriors directly.
    predictions = q.estimator_.predict(X[400:]) if Quantifier is CC else proba

    with config_context(prevalence_return_type="array"):
        out = np.asarray(q.aggregate(predictions, classes=[0, 1, 2]), dtype=float)

    _assert_full_prevalence(out, n_classes=3)


# --------------------------------------------------------------------------- #
# ``predict`` always spans every fitted class (exercises each aggregate path)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("Quantifier", MULTICLASS_QUANTIFIERS)
def test_predict_full_prevalence_multiclass(Quantifier):
    X, y = make_classification(
        n_samples=600, n_classes=3, n_informative=6, n_redundant=0,
        n_clusters_per_class=1, random_state=0,
    )
    q = _make(Quantifier).fit(X[:400], y[:400])
    with config_context(prevalence_return_type="array"):
        out = q.predict(X[400:])
    _assert_full_prevalence(out, n_classes=3)


@pytest.mark.parametrize("Quantifier", BINARY_QUANTIFIERS)
def test_predict_full_prevalence_binary_matching(Quantifier):
    X, y = make_classification(n_samples=500, n_classes=2, random_state=0)
    q = _make(Quantifier).fit(X[:350], y[:350])
    with config_context(prevalence_return_type="array"):
        out = q.predict(X[350:])
    _assert_full_prevalence(out, n_classes=2)
