import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlquantify._config import config_context
from mlquantify.likelihood import CDE, EMQ, MLPE


def _assert_valid_prevalence(prevalence, n_classes):
    prevalence = np.asarray(prevalence, dtype=float)
    assert prevalence.shape == (n_classes,)
    assert np.all(prevalence >= 0)
    assert np.all(prevalence <= 1)
    assert prevalence.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("quantifier_class", [EMQ, CDE])
def test_likelihood_methods_fit_predict_binary(quantifier_class, binary_dataset):
    X, y = binary_dataset
    q = quantifier_class(estimator=LogisticRegression(max_iter=1000, random_state=42))

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)


def test_mlpe_takes_no_estimator_and_trains_nothing(multiclass_dataset):
    """MLPE is non-aggregative: it takes no estimator and fits nothing.

    The estimate is the training prevalence, read from ``y``. There is no
    posterior to compute, so there is no estimator to fit -- and none to
    keep in ``estimator_``.
    """
    X, y = multiclass_dataset
    q = MLPE()
    q.fit(X, y)

    assert not hasattr(q, "estimator_")
    assert q._uses_estimator_predictions() is False

    with config_context(prevalence_return_type="array"):
        prevalence = np.asarray(q.predict(X), dtype=float)

    _assert_valid_prevalence(prevalence, n_classes=len(np.unique(y)))
    expected = np.asarray([np.mean(y == cls) for cls in np.unique(y)], dtype=float)
    np.testing.assert_allclose(prevalence, expected)


def test_mlpe_rejects_an_estimator():
    """The estimator is gone from the signature, not accepted and ignored.

    Accepting one would advertise work that never happens.
    """
    with pytest.raises(TypeError):
        MLPE(LogisticRegression(max_iter=1000, random_state=42))


def test_mlpe_ignores_the_test_sample(multiclass_dataset):
    """Any test sample yields the same answer: the training prevalence."""
    X, y = multiclass_dataset
    q = MLPE().fit(X, y)

    with config_context(prevalence_return_type="array"):
        dez = np.asarray(q.predict(X[:10]), dtype=float)
        todas = np.asarray(q.predict(X), dtype=float)

    np.testing.assert_allclose(dez, todas)


def test_emq_multiclass(multiclass_dataset):
    X, y = multiclass_dataset
    q = EMQ(estimator=LogisticRegression(max_iter=1000, random_state=42))

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=3)


@pytest.mark.parametrize("calib_function", ["bcts", "ts", "vs", "nbvs", None])
def test_emq_calibration(binary_dataset, calib_function):
    X, y = binary_dataset
    q = EMQ(
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        calib_function=calib_function,
    )

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)


def test_likelihood_dict_output_config(binary_dataset):
    X, y = binary_dataset
    q = EMQ(estimator=LogisticRegression(max_iter=1000, random_state=42))

    q.fit(X, y)

    with config_context(prevalence_return_type="dict"):
        prevalence = q.predict(X)

    assert set(prevalence) == {0, 1}
    assert sum(prevalence.values()) == pytest.approx(1.0)
