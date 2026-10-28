# likelihood/_generalized.py

import numpy as np

from mlquantify.base_aggregative import (
    AggregativeMixin,
    SoftPredictionMixin,
    _get_estimator,
    _get_estimator_function,
)
from mlquantify.compose import LikelihoodComposeQuantifier
from mlquantify.representations import PredictionRepresentation
from mlquantify.utils._validation import validate_prevalences


class MLPE(SoftPredictionMixin, AggregativeMixin, LikelihoodComposeQuantifier):
    r"""Maximum Likelihood Prevalence Estimation (MLPE) quantifier.

    The trivial quantification baseline. Under the assumption of *no* shift, the
    maximum-likelihood estimate of the test prevalence is exactly the observed
    training prevalence, so MLPE ignores the test set entirely and always
    returns the training class proportions. It is the reference lower bound that
    any genuine quantifier should beat; if a method cannot improve on MLPE, it
    is not exploiting the test data.

    Parameters
    ----------
    estimator : estimator, optional
        Kept for API compatibility with the aggregative quantifiers; it does
        **not** influence the estimate, since MLPE returns the training
        prevalence regardless of the test input.
    solver, tau_0, tau_1 : optional
        Retained for interface compatibility with the likelihood-composition
        framework; unused by this trivial baseline.
    cv : int or None, default=None
        Cross-validation folds used when fitting the (unused) estimator.
    stratified : bool, default=True
        Whether to stratify CV splits.
    shuffle : bool, default=False
        Whether to shuffle data before splitting.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.
    train_priors_ : ndarray of shape (n_classes,)
        Training class prevalences, returned for any test set.

    Notes
    -----
    MLPE makes no use of the test data; it is included only as a sanity-check
    baseline. Contrast with :class:`EMQ`, the non-trivial maximum-likelihood
    quantifier that re-weights the posteriors to the test set.

    See Also
    --------
    EMQ : Non-trivial maximum-likelihood quantifier (EM re-weighting).
    CC : Classify-and-count baseline that does use the test data.

    Examples
    --------
    >>> from mlquantify.likelihood import MLPE
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 4))
    >>> y = (rng.random(200) < 0.3).astype(int)
    >>> q = MLPE(LogisticRegression()).fit(X, y)
    >>> q.predict(rng.standard_normal((50, 4)))  # returns the training prevalence
    {0: ..., 1: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Saerens, M., Latinne, P., & Decaestecker, C. (2002).
               Adjusting the Outputs of a Classifier to New a Priori Probabilities.
               *Neural Computation*, 14(1), 2141–2156.
    """


    def __init__(
        self,
        estimator=None,
        solver="slsqp",
        tau_0=0.0,
        tau_1=0.0,
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            representation=PredictionRepresentation(
                method="soft",
                average=False,
            ),
            solver=solver,
            aggregative=True,
            tau_0=tau_0,
            tau_1=tau_1,
            random_state=random_state,
        )
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state

    def _fit_estimator_predictions(
        self,
        X,
        y,
        estimator_fitted=False,
        cv=None,
        stratified=None,
        shuffle=None,
        random_state=None,
        cv_prediction="refit",
    ):
        """Fit the estimator once, without cross-validation.

        The aggregative path cross-validates the estimator to obtain
        out-of-sample predictions, because an aggregation function trained on
        in-sample posteriors would be optimistically biased. MLPE has no
        aggregation function to train: the estimate is the training
        prevalence, read from ``y``. The cross-validation was therefore pure
        cost -- at the default ``cv`` it fit the estimator six times (five
        folds plus the refit) and called ``predict_proba`` five times, and
        then the estimate ignored all of it.

        A single fit is kept so that ``estimator_`` is still there after
        ``fit``, as it is for every other aggregative quantifier.

        ``cv``, ``stratified``, ``shuffle`` and the local ``random_state``
        are accepted and ignored, which is what the class docstring already
        said of the estimator itself.
        """
        estimator = _get_estimator(self)
        estimator_function = _get_estimator_function(self)

        if not estimator_fitted:
            # Fit in place, as the refit of the cross-validated path does.
            estimator = estimator.fit(X, y)
        self.estimator_ = estimator

        return getattr(estimator, estimator_function)(X), y

    def predict(self, X):
        """Return the training prevalence, ignoring ``X``.

        MLPE is the trivial baseline: it estimates the test prevalence as the
        observed training prevalence, so the test features are not used.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix (ignored).

        Returns
        -------
        prevalences : dict or ndarray of shape (n_classes,)
            The training class prevalences.
        """
        return validate_prevalences(self, self.train_priors_, self.classes_)

    def aggregate(
        self,
        test_representation,
        train_representation=None,
        train_labels=None,
        classes=None,
    ):
        """Return the training prevalence, ignoring the test representation.

        If ``train_labels`` is provided its class prevalence is used; otherwise
        the prevalence observed at ``fit`` time (``train_priors_``) is returned.
        """
        if train_labels is not None:
            train_labels = np.asarray(train_labels)
            cls = (
                np.asarray(classes)
                if classes is not None
                else np.unique(train_labels)
            )
            priors = np.asarray(
                [np.mean(train_labels == c) for c in cls],
                dtype=float,
            )
            self.classes_ = cls
        else:
            priors = np.asarray(self.train_priors_, dtype=float)

        return validate_prevalences(self, priors, self.classes_)
