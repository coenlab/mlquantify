# likelihood/_generalized.py

import numpy as np

from mlquantify.compose import LikelihoodComposeQuantifier
from mlquantify.utils._validation import validate_prevalences


class MLPE(LikelihoodComposeQuantifier):
    r"""Maximum Likelihood Prevalence Estimation (MLPE) quantifier.

    The trivial quantification baseline. Under the assumption of *no* shift, the
    maximum-likelihood estimate of the test prevalence is exactly the observed
    training prevalence, so MLPE ignores the test set entirely and always
    returns the training class proportions. It is the reference lower bound that
    any genuine quantifier should beat; if a method cannot improve on MLPE, it
    is not exploiting the test data.

    It is **not** aggregative and takes no estimator. ``fit`` reads the class
    proportions from ``y``; nothing is trained, and no posterior is ever
    computed. Taking a classifier would be taking one in order to never use
    it: the estimate does not depend on it, and fitting it was the whole cost
    of the method.

    Parameters
    ----------
    solver, tau_0, tau_1 : optional
        Retained for interface compatibility with the likelihood-composition
        framework; unused by this trivial baseline.
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

    Being non-aggregative, it has no ``estimator_`` after ``fit``, and
    ``cv``, ``stratified`` and ``shuffle`` are gone from the signature: there
    is no out-of-sample prediction to configure.

    See Also
    --------
    EMQ : Non-trivial maximum-likelihood quantifier (EM re-weighting).
    CC : Classify-and-count baseline that does use the test data.

    Examples
    --------
    >>> from mlquantify.likelihood import MLPE
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 4))
    >>> y = (rng.random(200) < 0.3).astype(int)
    >>> q = MLPE().fit(X, y)
    >>> q.predict(rng.standard_normal((50, 4)))  # returns the training prevalence
    {0: ..., 1: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Saerens, M., Latinne, P., & Decaestecker, C. (2002).
               Adjusting the Outputs of a Classifier to New a Priori Probabilities.
               *Neural Computation*, 14(1), 2141–2156.
    """


    # Keyword-only on purpose. With positional parameters, a caller written
    # for the old signature -- ``MLPE(LogisticRegression())`` -- would bind the
    # classifier to ``solver`` and run on silently misconfigured. A TypeError
    # is the right answer to asking for an estimator this method no longer has.
    def __init__(
        self,
        *,
        solver="slsqp",
        tau_0=0.0,
        tau_1=0.0,
        random_state=None,
    ):
        super().__init__(
            representation=None,
            estimator=None,
            solver=solver,
            aggregative=False,
            tau_0=tau_0,
            tau_1=tau_1,
            random_state=random_state,
        )
        self.random_state = random_state

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
