import numpy as np

from ._base import BaseRepresentation


class HistogramRepresentation(BaseRepresentation):
    r"""Histogram-based representation.
    
    This representation computes histograms for each feature (or selected subset) independently, and concatenates the bin frequencies into a single vector. When ``partition_blocks=True``, the attribute ``block_slices_`` is populated with the slice objects that identify each bin group in the concatenated output.

    Parameters
    ----------
    bins : int or array-like of shape (n_features,), default=(10,)
        The number of bins for each feature. If an integer, the same number of bins is used for all features.
    range : tuple of shape (2,), default=(0.0, 1.0)
        The lower and upper bounds for the histogram range.
    mode : str, default="histogram"
        The mode of the histogram. Options are:

        - "histogram": Compute the histogram counts.
        - "onehot": Compute a one-hot encoding of the histogram. In this mode, the output is a binary vector indicating which bin each value falls into, averaged over all samples.
    features : array-like of shape (n_features,), default=None
        The indices of the features to use. If None, all features are used to compute the histogram representation.
    partition_blocks : bool, default=False
        Whether to partition the output into contiguous blocks corresponding to each
        feature's bins. When ``partition_blocks=True``, the returned representation is
        still a single 1-D concatenated vector, but it is organized in feature-wise blocks:
        all bins for the first selected feature appear first, then all bins for the
        second selected feature, etc. In this case the attribute ``block_slices_`` is
        populated with a tuple of slice objects that identify the start/stop indices for
        each block, allowing easy extraction of per-feature bin groups from the
        concatenated vector. When ``partition_blocks=False``, the same concatenation is
        returned but no ``block_slices_`` metadata is stored.

        Example
        -------
        Suppose two features are selected and the bins parameter is (3, 4). The
        concatenated representation will have 3 + 4 = 7 entries. With
        ``partition_blocks=True``, ``block_slices_`` will be something like
        ``(slice(0, 3), slice(3, 7))``. To extract the second feature's block from a
        representation ``rep`` you can use ``rep[block_slices_[1]]`` which returns the
        4-element histogram for that feature.
    bin_edges : {'fixed', 'auto'}, default='fixed'
        Where the bin edges are placed.

        - ``'fixed'``: equal-width edges spanning ``range``.
        - ``'auto'``: per-feature edges fitted to the data in ``fit``, with
          the outer bins left unbounded so values beyond the fitted range
          fall in the end bins instead of being dropped. This is what
          histograms of raw features need, since features are not on the
          ``[0, 1]`` scale that ``range`` defaults to.
    laplace_smoothing : bool, default=False
        Whether to apply Laplace smoothing to the histogram counts.

    See Also
    --------
    KDERepresentation : Smooth (density) alternative to binning.
    PredictionRepresentation : Posterior/label representation.

    Examples
    --------
    >>> from mlquantify.representations._histogram import HistogramRepresentation
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> scores = rng.uniform(0, 1, (200, 1))
    >>> y = (scores[:, 0] > 0.5).astype(int)
    >>> rep = HistogramRepresentation(bins=(8,)).fit(scores, y)
    >>> rep.transform(scores[:10]).shape
    (8,)
    """

    def __init__(
        self,
        bins=(10,),
        range=(0.0, 1.0),
        mode="histogram",
        features=None,
        partition_blocks=False,
        bin_edges="fixed",
        laplace_smoothing=False,
    ):
        self.bins = np.atleast_1d(bins)
        self.range = range
        self.mode = mode
        self.features = features
        self.partition_blocks = partition_blocks
        self.bin_edges = bin_edges
        self.laplace_smoothing = laplace_smoothing

    def transform(self, X):
        """Compute the histogram representation for a set of instances.

        Each feature (or selected subset) is binned independently, and the
        bin frequencies are concatenated into a single vector.  When
        ``partition_blocks=True``, the attribute ``block_slices_`` is
        populated with the slice objects that identify each bin group in
        the concatenated output.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Feature matrix or 1-D score array.

        Returns
        -------
        representation : ndarray of shape (n_bins_total,)
            Normalized histogram vector (sums to 1 per feature-bin group).

        Examples
        --------
        >>> from mlquantify.representations._histogram import HistogramRepresentation
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> scores = rng.uniform(0, 1, (200, 1))
        >>> y = (scores[:, 0] > 0.5).astype(int)
        >>> rep = HistogramRepresentation(bins=(8,)).fit(scores, y)
        >>> rep.transform(scores[:10]).shape
        (8,)
        """
        X = self._as_2d(X)
        X = self._select_features(X)

        histograms = []
        block_slices = []
        start = 0

        for feature_idx in range(X.shape[1]):
            values = X[:, feature_idx]

            for bin_idx, n_bins in enumerate(self.bins):
                hist = self._compute_histogram(
                    values,
                    int(n_bins),
                    edges=self._get_edges(feature_idx, bin_idx),
                )
                histograms.append(hist)

                stop = start + hist.shape[0]
                block_slices.append(slice(start, stop))
                start = stop

        representation = np.concatenate(histograms)

        if self.partition_blocks:
            self.block_slices_ = tuple(block_slices)

        return representation

    def _fit(self, X, y, sample_weight=None):
        X = self._as_2d(X)
        X_selected = self._select_features(X)

        self._fit_edges(X_selected)

        self.class_representations_ = np.asarray([
            self.transform(X[y == cls])
            for cls in self.classes_
        ])

    def _compute_histogram(self, values, bins, edges=None):
        if self.mode not in {"histogram", "onehot"}:
            raise ValueError(f"Unknown mode: {self.mode!r}")

        if edges is not None:
            hist_edges = np.asarray(edges, dtype=float).copy()
            hist_edges[0] = -np.inf
            hist_edges[-1] = np.inf
            hist, _ = np.histogram(
                values,
                bins=hist_edges,
                density=False,
            )

            hist = hist.astype(float)
            return self._normalize(hist)

        if self.mode == "histogram":
            hist, _ = np.histogram(
                values,
                bins=bins,
                range=self.range,
                density=False,
            )

            hist = hist.astype(float)
            return self._normalize(hist)

        if self.mode == "onehot":
            edges = np.linspace(
                self.range[0],
                self.range[1],
                bins + 1,
            )

            indices = np.digitize(values, edges[1:-1], right=False)

            onehot = np.eye(bins)[indices]

            return onehot.mean(axis=0)
        raise ValueError(f"Unknown mode: {self.mode!r}")

    def _normalize(self, hist):
        n = max(hist.sum(), 1.0)
        if self.laplace_smoothing:
            k = hist.shape[0]
            return (hist + 1.0 / k) / (n + 1)
        return hist / n

    def _fit_edges(self, X):
        if self.bin_edges == "fixed":
            return

        if self.bin_edges != "auto":
            raise ValueError("bin_edges must be either 'fixed' or 'auto'.")

        self.edges_ = []

        for feature_idx in range(X.shape[1]):
            values = X[:, feature_idx]
            feature_edges = []

            for n_bins in self.bins:
                feature_edges.append(
                    np.histogram_bin_edges(values, bins=int(n_bins))
                )

            self.edges_.append(feature_edges)

    def _get_edges(self, feature_idx, bin_idx):
        if self.bin_edges == "fixed":
            return None

        if self.bin_edges != "auto":
            raise ValueError("bin_edges must be either 'fixed' or 'auto'.")

        if not hasattr(self, "edges_"):
            raise AttributeError(
                "HistogramRepresentation must be fitted before transform() "
                "when bin_edges='auto'."
            )

        return self.edges_[feature_idx][bin_idx]

    def _select_features(self, X):
        if self.features is None:
            return X

        if isinstance(self.features, (int, np.integer)):
            feature = int(self.features)
            if X.shape[1] == 1 and feature == 1:
                return X
            return X[:, [feature]]

        return X[:, self.features]

    @staticmethod
    def _as_2d(X):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            return X.reshape(-1, 1)

        return X
