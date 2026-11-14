import numpy as np
from mlquantify.utils._constraints import Interval, Options
from mlquantify.neighbors._classification import PWKCLF
from mlquantify.utils import validate_data
from mlquantify.counting import CC, ACC


class PWK(CC):
    r"""Probabilistic Weighted k-Nearest Neighbour (PWK) quantifier.

    Targets prior probability shift. PWK is an **aggregative** Classify-and-Count
    quantifier — it shares the standard ``fit`` / ``predict`` / ``aggregate``
    interface of :class:`~mlquantify.counting.CC` — but its classifier is a
    k-nearest-neighbour rule *modified for quantification* (:class:`PWKCLF`):
    each neighbour's vote is re-weighted by a class-specific factor (controlled
    by ``alpha``) so the count is not dominated by the majority class. Unlike the
    other aggregative quantifiers, PWK therefore takes **no external estimator
    parameter**: the modified k-NN is intrinsic to the method.

    Parameters
    ----------
    alpha : float, default=1
        Imbalance-correction exponent. ``1`` applies the standard inverse-size
        weighting; higher values further amplify minority-class neighbours.
    n_neighbors : int, default=10
        Number of nearest neighbours considered for each test instance.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Neighbour-search algorithm.

        - ``'auto'`` : pick the best of the below from the fitted data (default).
        - ``'ball_tree'`` : ball-tree index; good in higher dimensions.
        - ``'kd_tree'`` : k-d tree index; fast in low dimensions.
        - ``'brute'`` : exhaustive search; exact, best for small data.
    metric : str, default='euclidean'
        Distance metric for the neighbour search.
    leaf_size : int, default=30
        Leaf size for the tree-based algorithms (speed/memory trade-off).
    p : int, default=2
        Power parameter for the Minkowski metric (``1`` = Manhattan, ``2`` = Euclidean).
    metric_params : dict or None, default=None
        Additional keyword arguments for the metric function.
    n_jobs : int or None, default=None
        Number of parallel jobs for the neighbour search.

    Attributes
    ----------
    estimator : PWKCLF
        The underlying weighted k-NN classifier (built from the parameters
        above; not an argument).
    estimator_ : PWKCLF
        The fitted classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    PWK is a classify-and-count method whose only quantification-specific
    ingredient is the imbalance re-weighting; it needs no separate scorer and
    handles multiclass directly, but inherits k-NN's sensitivity to feature
    scaling and dimensionality. Because it subclasses :class:`CC`, ``aggregate``
    (with its optional ``classes`` argument) is available too.

    See Also
    --------
    CC : Plain classify-and-count baseline.
    ACC : Adjusted count for binary prior shift.

    Examples
    --------
    >>> from mlquantify.neighbors import PWK
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = PWK(alpha=1.5, n_neighbors=5).fit(X, y)
    >>> q.predict(X)
    {0: ..., 1: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Barranquero, J., González, P., Díez, J., & del Coz, J. J. (2013).
               On the study of nearest neighbor algorithms for prevalence
               estimation in binary problems.
               *Pattern Recognition*, 46(2), 472–482.
    """

    _parameter_constraints = {
        "alpha": [Interval(1, None, inclusive_right=False)],
        "n_neighbors": [Interval(1, None, inclusive_right=False)],
        "algorithm": [Options(["auto", "ball_tree", "kd_tree", "brute"])],
        "metric": [str],
        "leaf_size": [Interval(1, None, inclusive_right=False)],
        "p": [Interval(1, None, inclusive_right=False)],
        "metric_params": [dict, type(None)],
        "n_jobs": [Interval(-1, None, inclusive_right=False), type(None)],
    }

    def __init__(self,
                 alpha=1,
                 n_neighbors=10,
                 algorithm="auto",
                 metric="euclidean",
                 leaf_size=30,
                 p=2,
                 metric_params=None,
                 n_jobs=None):
        self.alpha = alpha
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.leaf_size = leaf_size
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        # PWK is aggregative, but its estimator is fixed to the quantification
        # k-NN rather than supplied by the user.
        super().__init__(
            estimator=PWKCLF(alpha=alpha,
                             n_neighbors=n_neighbors,
                             algorithm=algorithm,
                             metric=metric,
                             leaf_size=leaf_size,
                             p=p,
                             metric_params=metric_params,
                             n_jobs=n_jobs),
        )

    def classify(self, X):
        """Classify test instances using the underlying weighted k-NN estimator.

        Returns hard class labels produced by :class:`PWKCLF` without any
        prevalence-level aggregation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted class label for each test instance.

        Examples
        --------
        >>> from mlquantify.neighbors import PWK
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, random_state=42)
        >>> q = PWK(alpha=1.5, n_neighbors=5).fit(X, y)
        >>> labels = q.classify(X[:5])
        """
        X = validate_data(self, X, ensure_2d=True)
        return self.estimator_.predict(X)


class PWKAC(ACC):
    r"""Adjusted Probabilistic Weighted k-Nearest Neighbour (PWKAC) quantifier.

    Targets prior probability shift. This is the quantifier Barranquero et al.
    propose: the weighted k-NN rule of :class:`PWKCLF` inside the adjusted
    count, with ``TPR`` and ``FPR`` estimated by **leave-one-out** on the
    training set. Their argument for nearest neighbours in quantification is
    exactly this estimate: the adjustment is only as good as the rates it
    corrects by, and k-NN is the one learner that can afford leave-one-out,
    because the neighbour query is made once and every fold reads from it.
    :class:`PWK`, in the same module, is the same classifier under plain
    classify-and-count, without the correction.

    Like :class:`~mlquantify.counting.ACC`, this is a **binary-only** method;
    multiclass problems are decomposed one-vs-rest, and each sub-problem gets
    its own class weights and its own leave-one-out estimate.

    Parameters
    ----------
    alpha : float, default=1
        Imbalance-correction exponent of the weighted k-NN rule. ``1`` is the
        value of the original study.
    n_neighbors : int, default=10
        Number of nearest neighbours considered for each instance.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Neighbour-search algorithm.
    metric : str, default='euclidean'
        Distance metric for the neighbour search.
    leaf_size : int, default=30
        Leaf size for the tree-based algorithms (speed/memory trade-off).
    p : int, default=2
        Power parameter for the Minkowski metric (``1`` = Manhattan,
        ``2`` = Euclidean).
    metric_params : dict or None, default=None
        Additional keyword arguments for the metric function.
    n_jobs : int or None, default=None
        Number of parallel jobs for the neighbour search and for the
        one-vs-rest decomposition.
    strategy : {'ovr', 'ovo'}, default='ovr'
        Multiclass decomposition strategy.

    Attributes
    ----------
    estimator : PWKCLF
        The underlying weighted k-NN classifier (built from the parameters
        above; not an argument).
    estimator_ : PWKCLF
        The fitted classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.
    train_predictions : ndarray of shape (n_samples,)
        Leave-one-out labels for the training set, from which ``TPR`` and
        ``FPR`` are read.
    y_train : ndarray of shape (n_samples,)
        True labels corresponding to ``train_predictions``.

    Notes
    -----
    Why leave-one-out, and not cross-validation or the training predictions
    themselves:

    * Predicting the training set with the fitted rule is resubstitution: a
      training point is its own nearest neighbour at distance zero, and under
      the PWK weights that self-vote carries the largest weight of all when
      the point belongs to the minority class. The resulting rates are
      optimistic and the adjustment they produce is too small. The bias grows
      with the difficulty of the problem, which is where the correction
      matters most.
    * k-fold cross-validation is unbiased but estimates the rates of a
      classifier trained on a fraction of the data, and costs k neighbour
      structures.
    * Leave-one-out estimates the rates of the deployed classifier and costs
      one neighbour query: ask for ``n_neighbors + 1`` neighbours of each
      training point and drop the first, which is the point itself.

    The usual caveat of that trick applies: with exact duplicates in the
    training set the dropped neighbour may be a copy rather than the point
    itself. The vote then still excludes one of the identical points, so the
    effect is limited to that duplicate's label.

    See Also
    --------
    PWK : The same classifier under plain classify-and-count.
    ACC : Adjusted count over a user-supplied classifier, with the rates
        estimated by cross-validation.

    Examples
    --------
    >>> from mlquantify.neighbors import PWKAC
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = PWKAC(n_neighbors=10).fit(X, y)
    >>> q.predict(X)
    {0: ..., 1: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Barranquero, J., Gonzalez, P., Diez, J., & del Coz, J. J. (2013).
               On the study of nearest neighbor algorithms for prevalence
               estimation in binary problems.
               *Pattern Recognition*, 46(2), 472-482.
        .. [2] Forman, G. (2008). Quantifying Counts and Costs via
               Classification. *Data Mining and Knowledge Discovery*,
               17(2), 164-206.
    """

    _parameter_constraints = {
        "alpha": [Interval(1, None, inclusive_right=False)],
        "n_neighbors": [Interval(1, None, inclusive_right=False)],
        "algorithm": [Options(["auto", "ball_tree", "kd_tree", "brute"])],
        "metric": [str],
        "leaf_size": [Interval(1, None, inclusive_right=False)],
        "p": [Interval(1, None, inclusive_right=False)],
        "metric_params": [dict, type(None)],
        "n_jobs": [Interval(-1, None, inclusive_right=False), type(None)],
        "strategy": [Options(["ovr", "ovo"])],
    }

    def __init__(self,
                 alpha=1,
                 n_neighbors=10,
                 algorithm="auto",
                 metric="euclidean",
                 leaf_size=30,
                 p=2,
                 metric_params=None,
                 n_jobs=None,
                 strategy="ovr"):
        self.alpha = alpha
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.leaf_size = leaf_size
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        # Like PWK, the estimator is fixed to the quantification k-NN instead
        # of being supplied by the user. The cross-validation parameters of
        # ACC stay at their defaults and are never used: the rates come from
        # leave-one-out.
        super().__init__(
            estimator=PWKCLF(alpha=alpha,
                             n_neighbors=n_neighbors,
                             algorithm=algorithm,
                             metric=metric,
                             leaf_size=leaf_size,
                             p=p,
                             metric_params=metric_params,
                             n_jobs=n_jobs),
            strategy=strategy,
            n_jobs=n_jobs,
        )

    def _fit_estimator_predictions(self, X, y, estimator_fitted=False,
                                   cv=None, stratified=None, shuffle=None,
                                   random_state=None, cv_prediction="refit"):
        """Fit the weighted k-NN and label the training set leave-one-out.

        Replaces the cross-validation of
        :class:`~mlquantify.counting._base.BaseAdjustCount`; the
        cross-validation arguments are accepted and ignored.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if estimator_fitted:
            self.estimator_ = self.estimator
            return self.estimator_.predict(X), y
        self.estimator_ = self.estimator.fit(X, y)
        return self._leave_one_out(X), y

    def _leave_one_out(self, X):
        """Label each training point with the neighbours that are not itself."""
        estimator = self.estimator_
        n_samples = X.shape[0]
        classes = estimator.classes_
        # One query for the whole training set: ask for one neighbour more
        # than the rule uses and drop the first column, which is the point
        # itself at distance zero.
        n_query = min(estimator.nbrs.n_neighbors + 1, n_samples)
        indices = estimator.nbrs.kneighbors(
            X, n_neighbors=n_query, return_distance=False)[:, 1:]
        if indices.shape[1] == 0:
            # A single training point has no other neighbour to vote.
            return np.repeat(classes[0], n_samples)

        labels = np.asarray(estimator.y_train)[indices]
        codes = np.searchsorted(classes, labels)
        counts = np.zeros((n_samples, len(classes)), dtype=float)
        np.add.at(counts, (np.arange(n_samples)[:, None], codes), 1.0)
        counts *= estimator.class_weights
        return classes[np.argmax(counts, axis=1)]
