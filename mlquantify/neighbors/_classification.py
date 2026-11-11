
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors



class PWKCLF(ClassifierMixin, BaseEstimator):
    r"""Probabilistic Weighted k-Nearest Neighbor Classifier (PWKCLF).

    A weighted k-nearest neighbor classifier that assigns class probabilities to 
    instances based on neighbor counts weighted by class-specific inverse frequency 
    factors adjusted by a hyperparameter alpha controlling imbalance compensation. 

    Attributes
    ----------
    alpha : float
        Exponent controlling the degree of imbalance compensation.
    n_neighbors : int
        Number of nearest neighbors considered.
    nbrs : sklearn.neighbors.NearestNeighbors
        The underlying k-NN structure used for neighbor queries.
    classes_ : ndarray
        Unique classes observed during training.
    class_to_index : dict
        Mapping from class label to index used in internal arrays.
    class_weights : ndarray
        Per-class weights computed based on class frequency and alpha.
    y_train : ndarray
        Labels of training samples.


    Notes
    -----
    The class weights are defined as:

    .. math::

        w_c = \left( \frac{N_c}{\min_{c'} N_{c'}} \right)^{-\frac{1}{\alpha}},

    where :math:`N_c` is the count of class :math:`c` in the training set.

    This weighting scheme reduces bias towards majority classes by downweighting them
    in the voting process.

    Examples
    --------
    >>> import numpy as np
    >>> X_train, y_train = np.random.randn(60, 4), np.random.randint(0, 2, 60)
    >>> X_test = np.random.randn(10, 4)
    >>> clf = PWKCLF(alpha=2.0, n_neighbors=7)
    >>> _ = clf.fit(X_train, y_train)
    >>> labels = clf.predict(X_test)
    """
    
    def __init__(self,
                 alpha=1,
                 n_neighbors=10,
                 algorithm="auto",
                 metric="euclidean",
                 leaf_size=30,
                 p=2,
                 metric_params=None,
                 n_jobs=None):
        # Only store the constructor arguments here. ``get_params`` reads them
        # back by name, and ``sklearn.base.clone`` rebuilds the estimator from
        # them -- so every argument must survive as an attribute of the same
        # name, and no work may happen before ``fit``.
        self.alpha = alpha
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.leaf_size = leaf_size
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        self.classes_ = None
        self.class_to_index = None
        self.class_weights = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fit the PWKCLF model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.

        y : array-like of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : object
            The fitted instance.
        """
        n_samples = X.shape[0]
        # Built here, not in ``__init__``: a scikit-learn estimator must do no
        # work before ``fit``, and the neighbour count depends on the training
        # set when it is smaller than ``n_neighbors``.
        self.nbrs = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, n_samples),
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs)

        self.y_train = y
        
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.classes_ = unique_classes
        self.class_to_index = dict(zip(self.classes_, range(len(self.classes_))))

        min_class_count = np.min(class_counts)
        self.class_weights = (class_counts / min_class_count) ** (-1.0 / self.alpha)
        self.nbrs.fit(X)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to predict.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        n_samples = X.shape[0]
        nn_indices = self.nbrs.kneighbors(X, return_distance=False)

        CM = np.zeros((n_samples, len(self.classes_)))
        
        for i in range(n_samples):
            for j in nn_indices[i]:
                CM[i, self.class_to_index[self.y_train[j]]] += 1

        CM = np.multiply(CM, self.class_weights)
        predictions = np.apply_along_axis(np.argmax, axis=1, arr=CM)
        
        return self.classes_[predictions]
