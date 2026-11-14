
import pytest
import numpy as np
from mlquantify.neighbors import PWK, PWKAC
from mlquantify.neighbors._classification import PWKCLF


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


# ---------------------------------------------------------------------------
# PWKAC: the adjusted count over the weighted k-NN, with leave-one-out rates
# ---------------------------------------------------------------------------

def test_pwkac_fit_predict(binary_dataset):
    X, y = binary_dataset
    preds = PWKAC(n_neighbors=10).fit(X, y).predict(X)
    _assert_valid_prevalence(preds, n_classes=2)


def test_pwkac_multiclass(multiclass_dataset):
    X, y = multiclass_dataset
    preds = PWKAC(n_neighbors=10).fit(X, y).predict(X)
    _assert_valid_prevalence(preds, n_classes=3)


def test_pwkac_train_predictions_are_leave_one_out(binary_dataset):
    """The stored training labels must not be the resubstitution ones.

    Resubstitution lets each point vote for itself, which is what makes the
    rates optimistic; leave-one-out drops that vote. On data where the rule
    makes mistakes the two labelings have to differ.
    """
    X, y = binary_dataset
    q = PWKAC(n_neighbors=10).fit(X, y)
    reinsercao = q.estimator_.predict(X)
    assert q.train_predictions.shape == y.shape
    assert not np.array_equal(q.train_predictions, reinsercao)
    acerto_loo = np.mean(q.train_predictions == y)
    acerto_reinsercao = np.mean(reinsercao == y)
    assert acerto_loo <= acerto_reinsercao


def test_pwkac_leave_one_out_matches_an_explicit_loop(binary_dataset):
    """The single-query trick must agree with refitting without each point."""
    X, y = binary_dataset
    X, y = X[:60], y[:60]
    q = PWKAC(n_neighbors=5).fit(X, y)

    esperado = np.empty_like(y)
    for i in range(len(X)):
        resto = np.ones(len(X), dtype=bool)
        resto[i] = False
        clf = PWKCLF(alpha=1, n_neighbors=5).fit(X[resto], y[resto])
        esperado[i] = clf.predict(X[i:i + 1])[0]

    # Leaving a point out also changes the class weights of the rule, by one
    # example out of sixty, so a handful of borderline points may flip. The
    # bulk has to agree.
    assert np.mean(q.train_predictions == esperado) > 0.95


def test_pwkac_corrects_the_bias_of_pwk():
    """On an imbalanced problem the adjustment must beat the plain count."""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=400, n_features=10, n_informative=5,
                               weights=[0.25, 0.75], random_state=42)
    verdade = np.bincount(y)[1] / len(y)
    contagem = np.asarray(PWK(n_neighbors=10).fit(X, y).predict(X))[1]
    ajustada = np.asarray(PWKAC(n_neighbors=10).fit(X, y).predict(X))[1]
    assert abs(ajustada - verdade) < abs(contagem - verdade)


def test_pwkac_params(binary_dataset):
    X, y = binary_dataset
    q = PWKAC(alpha=2, n_neighbors=7, metric="manhattan")
    assert q.get_params()["alpha"] == 2
    assert q.get_params()["n_neighbors"] == 7
    q.fit(X, y)
    assert q.estimator_.n_neighbors == 7
    _assert_valid_prevalence(q.predict(X), n_classes=2)


def test_pwkac_more_neighbors_than_samples():
    """``n_neighbors`` above the training size must not break the query."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    preds = PWKAC(n_neighbors=20).fit(X, y).predict(X)
    _assert_valid_prevalence(preds, n_classes=2)
