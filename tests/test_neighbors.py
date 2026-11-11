
import pytest
import numpy as np
from mlquantify.neighbors import PWK


def _assert_valid_prevalence(prevalence, n_classes):
    prevalence = np.asarray(prevalence, dtype=float)
    assert prevalence.shape == (n_classes,)
    assert np.all(prevalence >= 0)
    assert np.all(prevalence <= 1)
    assert prevalence.sum() == pytest.approx(1.0)


def test_pwk_fit_predict(binary_dataset):
    X, y = binary_dataset
    q = PWK(n_neighbors=10)
    q.fit(X, y)
    preds = q.predict(X)
    _assert_valid_prevalence(preds, n_classes=2)

def test_pwk_multiclass(multiclass_dataset):
    X, y = multiclass_dataset
    q = PWK(n_neighbors=10)
    q.fit(X, y)
    preds = q.predict(X)
    _assert_valid_prevalence(preds, n_classes=3)

def test_pwk_params(binary_dataset):
    X, y = binary_dataset
    q = PWK(n_neighbors=5, alpha=2.0)
    q.fit(X, y)
    preds = q.predict(X)
    _assert_valid_prevalence(preds, n_classes=2)

    # Test invalid n_neighbors
    with pytest.raises(ValueError):
         PWK(n_neighbors=-1).fit(X, y)
    
    # Test high neighbors count
    q = PWK(n_neighbors=len(X) - 1).fit(X, y)


# --------------------------------------------------------------------------
# PWKCLF as a scikit-learn estimator
#
# Both of these were broken: ``PWKCLF`` was not exported, and it had no
# ``get_params``, so ``sklearn.base.clone`` refused it. Together that meant a
# user could not build "PWK with count adjustment" out of the two pieces this
# library already ships -- ``ACC`` clones its estimator for cross-validation.
# --------------------------------------------------------------------------

def test_pwkclf_is_exported():
    from mlquantify.neighbors import PWKCLF  # noqa: F401

    import mlquantify.neighbors as mod

    assert "PWKCLF" in mod.__all__


def test_pwkclf_get_params_round_trips():
    from sklearn.base import clone

    from mlquantify.neighbors import PWKCLF

    clf = PWKCLF(alpha=2.0, n_neighbors=7, metric="manhattan", leaf_size=17)
    params = clf.get_params()
    assert params["alpha"] == 2.0
    assert params["n_neighbors"] == 7
    assert params["metric"] == "manhattan"
    assert params["leaf_size"] == 17

    copia = clone(clf)
    assert copia is not clf
    assert copia.get_params() == params


def test_pwkclf_fits_after_clone(binary_dataset):
    from sklearn.base import clone

    from mlquantify.neighbors import PWKCLF

    X, y = binary_dataset
    clf = clone(PWKCLF(n_neighbors=5))
    clf.fit(X, y)
    rotulos = clf.predict(X)
    assert rotulos.shape == (X.shape[0],)
    assert set(np.unique(rotulos)).issubset(set(np.unique(y)))


def test_pwkclf_works_as_estimator_of_acc(binary_dataset):
    """O conserto que importa: montar PWK com ajuste de contagem."""
    from mlquantify.counting import ACC
    from mlquantify.neighbors import PWKCLF

    X, y = binary_dataset
    q = ACC(estimator=PWKCLF(n_neighbors=5), cv=3)
    q.fit(X, y)
    _assert_valid_prevalence_dict(q.predict(X), n_classes=2)


def _assert_valid_prevalence_dict(prevalence, n_classes):
    if isinstance(prevalence, dict):
        prevalence = np.asarray([prevalence[k] for k in sorted(prevalence)])
    _assert_valid_prevalence(prevalence, n_classes)
