from __future__ import annotations
from typing import Any
import numpy as np
import scipy.sparse as sp
from sklearn.utils.validation import check_array, check_X_y, _check_y

from mlquantify.utils._tags import TargetInputTags, get_tags
from mlquantify.utils._exceptions import InputValidationError, InvalidParameterError, NotFittedError
from mlquantify.utils._constraints import make_constraint


# ---------------------------
# y Validation
# ---------------------------

def _validate_is_numpy_array(array: Any) -> None:
    """Ensure y are a numpy array."""
    if not isinstance(array, np.ndarray):
        raise InputValidationError(
            f"y must be a numpy array, got {type(y).__name__}."
        )


def _validate_1d_predictions(quantifier: Any, y: np.ndarray, target_tags: TargetInputTags) -> None:
    """Validate 1D predictions according to quantifier tags."""
    if target_tags.continuous:
        return  # continuous allows any numeric vector

    n_class = len(np.unique(y))

    if target_tags.one_d:
        
        if n_class > 2 and not target_tags.multi_class:
            raise InputValidationError(
                f"1D predictions for {quantifier.__class__.__name__} must be binary "
                f"with 2 unique values, got {n_class} unique values."
            )
        if not np.issubdtype(y.dtype, np.number) and not target_tags.categorical:
            raise InputValidationError(
                f"1D predictions for {quantifier.__class__.__name__} must be numeric (int or float), "
                f"got dtype {y.dtype}."
            )
        return

    if not target_tags.one_d:
        raise InputValidationError(
            f"{quantifier.__class__.__name__} does not accept 1D input according to its tags."
        )


def _validate_2d_predictions(quantifier: Any, y: np.ndarray, target_tags: TargetInputTags) -> None:
    """Validate 2D predictions according to quantifier tags."""
    if not (target_tags.two_d or target_tags.multi_class):
        raise InputValidationError(
            f"{quantifier.__class__.__name__} does not accept multi-class or 2D input."
        )

    if not np.issubdtype(y.dtype, np.floating):
        raise InputValidationError(
            f"{quantifier.__class__.__name__} expects float probabilities for 2D predictions, "
            f"got dtype {y.dtype}."
        )

    # Efficient normalization check for soft probabilities
    if target_tags.two_d:
        row_sums = y.sum(axis=1)
        if np.abs(row_sums - 1).max() > 1e-3:
            raise InputValidationError(
                f"Soft predictions for multiclass quantifiers must sum to 1 across columns "
                f"(max deviation={np.abs(row_sums - 1).max():.3g})."
            )


def validate_y(quantifier: Any, y: np.ndarray) -> None:
    """
    Validate predictions using the quantifier's declared input tags.
    Raises InputValidationError if inconsistent with tags.
    """
    _validate_is_numpy_array(y)

    try:
        tags = get_tags(quantifier)
        target_tags = tags.target_input_tags
    except AttributeError as e:
        raise InputValidationError(
            f"Quantifier {quantifier.__class__.__name__} does not implement __mlquantify_tags__()."
        ) from e
    if y.ndim == 1:
        _validate_1d_predictions(quantifier, y, target_tags)
    elif y.ndim == 2:
        _validate_2d_predictions(quantifier, y, target_tags)
    else:
        raise InputValidationError(
            f"Predictions must be 1D or 2D array, got array with ndim={y.ndim} and shape={y.shape}."
        )

def _get_valid_crisp_predictions(predictions, y_train=None, threshold=0.5):
    predictions = np.asarray(predictions)
    dimensions = predictions.ndim

    if y_train is not None:
        classes = np.unique(y_train)
    else:
        classes = None

    if dimensions > 2:
        # Assuming the last dimension contains class probabilities
        crisp_indices = np.argmax(predictions, axis=-1)
        if classes is not None:
            predictions = classes[crisp_indices]
        else:
            predictions = crisp_indices
    elif dimensions == 2:
        # Binary or multi-class probabilities (N, C)
        if classes is not None and len(classes) == 2:
            # Binary case with explicit classes
            predictions = np.where(predictions[:, 1] >= threshold, classes[1], classes[0])
        elif classes is not None and len(classes) > 2:
            # Multi-class case with explicit classes
            crisp_indices = np.argmax(predictions, axis=1)
            predictions = classes[crisp_indices]
        else:
            # Default binary (0 or 1) or multi-class (0 to C-1)
            if predictions.shape[1] == 2:
                predictions = (predictions[:, 1] >= threshold).astype(int)
            else:
                predictions = np.argmax(predictions, axis=1)
    elif dimensions == 1:
        # 1D probabilities (e.g., probability of positive class)
        if np.issubdtype(predictions.dtype, np.floating):
            if classes is not None and len(classes) == 2:
                predictions = np.where(predictions >= threshold, classes[1], classes[0])
            else:
                predictions = (predictions >= threshold).astype(int)
    else:
        raise ValueError(f"Predictions array has an invalid number of dimensions. Expected 1 or more dimensions, got {predictions.ndim}.")

    return predictions


def validate_predictions(quantifier: Any, predictions: np.ndarray, threshold: float = 0.5, y_train=None) -> np.ndarray:
    """
    Validate predictions using the quantifier's declared output tags.
    Raises InputValidationError if inconsistent with tags.
    """
    _validate_is_numpy_array(predictions)

    try:
        tags = get_tags(quantifier)
        estimator_type = tags.estimator_type
    except AttributeError as e:
        raise InputValidationError(
            f"Quantifier {quantifier.__class__.__name__} does not implement __mlquantify_tags__()."
        ) from e

    if estimator_type == "soft" and np.issubdtype(predictions.dtype, np.integer):
        raise InputValidationError(
            f"Soft predictions for {quantifier.__class__.__name__} must be float, got dtype {predictions.dtype}."
        )
    elif estimator_type == "crisp" and np.issubdtype(predictions.dtype, np.floating):
        predictions = _get_valid_crisp_predictions(predictions, y_train, threshold) 
    return predictions   
    
    


# ---------------------------
# Parameter Validation
# ---------------------------

def validate_parameter_constraints(parameter_constraints: dict[str, Any], params: dict[str, Any], caller_name: str) -> None:
    """Validate parameters against their declared constraints."""
    for param_name, param_val in params.items():
        if param_name not in parameter_constraints:
            continue

        constraints = parameter_constraints[param_name]
        if constraints == "no_validation":
            continue

        constraint_objs = [make_constraint(c) for c in constraints]

        if any(c.is_satisfied_by(param_val) for c in constraint_objs):
            continue  # valid parameter

        # Only visible constraints in error message
        visible = [c for c in constraint_objs if not getattr(c, "hidden", False)] or constraint_objs
        constraint_str = (
            str(visible[0])
            if len(visible) == 1
            else ", ".join(map(str, visible[:-1])) + f" or {visible[-1]}"
        )

        raise InvalidParameterError(
            f"The parameter '{param_name}' of {caller_name} must be {constraint_str}. "
            f"Got {param_val!r} (type={type(param_val).__name__})."
        )
        
        
def validate_estimator_constraints(quantifier, estimator) -> None:
    """Validate the estimator parameter of a quantifier."""
    try:
        tags = get_tags(quantifier)
    except AttributeError as e:
        raise InvalidParameterError(
            f"Quantifier {quantifier.__class__.__name__} does not implement __mlquantify_tags__()."
        ) from e

    if not tags.has_estimator:
        if estimator is not None:
            raise InvalidParameterError(
                f"The quantifier {quantifier.__class__.__name__} does not support using an estimator."
            )
        return

    estimator_function = tags.estimator_function
    if estimator_function is None:
        raise InvalidParameterError(
            f"The quantifier {quantifier.__class__.__name__} does not specify a valid estimator_function in its tags."
        )
    if estimator is None:
        raise InvalidParameterError(
            f"The quantifier {quantifier.__class__.__name__} requires an estimator."
        )
    if not hasattr(estimator, estimator_function):
        raise InvalidParameterError(
            f"The provided estimator does not have a '{estimator_function}' method, "
            f"which is required by the quantifier {quantifier.__class__.__name__}."
        )


def _is_fitted(quantifier, attributes=None, all_or_any=all):
    """Check if the quantifier is fitted by verifying the presence of specified attributes."""
    if attributes is None:
        attributes = ["is_fitted_"]

    checks = [hasattr(quantifier, attr) for attr in attributes]
    return all(checks) if all_or_any == all else any(checks)


def check_is_fitted(quantifier, attributes=None, *, msg=None, all_or_any=all):
    """Raise NotFittedError if the quantifier is not fitted."""

    if msg is None:
        msg = f"This {quantifier.__class__.__name__} instance is not fitted yet. Call 'fit' first."

    if not hasattr(quantifier, "fit"):
        raise TypeError(f"Cannot check if {quantifier.__class__.__name__} is fitted: no 'fit' method found.")
    
    tags = get_tags(quantifier)
    
    if not tags.requires_fit and attributes is None:
        return  # No fitting required for this quantifier
    
    if not _is_fitted(quantifier, attributes, all_or_any):
        raise NotFittedError(msg % {"name": type(quantifier).__name__})


def _is_arraylike_not_scalar(array):
    """Return True if array is array-like and not a scalar"""
    return _is_arraylike(array) and not np.isscalar(array)


def _is_arraylike(x):
    """Returns whether the input is array-like."""
    if sp.issparse(x):
        return False

    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")


def _transform_if_float(y):
    """Transform y to integers if it is float."""
    if np.issubdtype(y.dtype, np.floating):
        y = y.astype(str)
    return y


def validate_data(quantifier, 
                  X="no_validation",
                  y="no_validation",
                  reset=True,
                  validate_separately=False,
                  skip_check_array=False,
                  **check_params):
    """
    Validate input data X and optionally y using sklearn's validate_data.
    Raises InputValidationError if validation fails.
    """
    tags = get_tags(quantifier)
    if y is None and tags.target_input_tags.required:
        raise InputValidationError(
            f"The target variable y is required for {quantifier.__class__.__name__}."
        )
    
    no_val_X = isinstance(X, str) and X == "no_validation"
    no_val_y = y is None or (isinstance(y, str) and y == "no_validation")
    
    if no_val_X and no_val_y:
        raise ValueError("Validation should be done on X, y or both.")
    
    default_check_params = {"estimator": quantifier}
    check_params = {**default_check_params, **check_params}
    
    if skip_check_array:
        if not no_val_X and no_val_y:
            out = X
        elif no_val_X and not no_val_y:
            out = y
        else:
            out = X, y
    elif not no_val_X and no_val_y:
        out = check_array(X, input_name="X", dtype=None, **check_params)
    elif no_val_X and not no_val_y:
        out = _check_y(y, **check_params)
    else:
        if validate_separately:
            # We need this because some estimators validate X and y
            # separately, and in general, separately calling check_array()
            # on X and y isn't equivalent to just calling check_X_y()
            # :(
            check_X_params, check_y_params = validate_separately
            if "estimator" not in check_X_params:
                check_X_params = {**default_check_params, **check_X_params}
            X = check_array(X, input_name="X", dtype=None, **check_X_params)
            if "estimator" not in check_y_params:
                check_y_params = {**default_check_params, **check_y_params}
            y = check_array(y, input_name="y", **check_y_params)
            y = _transform_if_float(y)
        else:
            X, y = check_X_y(X, y, dtype=None, **check_params)
            y = _transform_if_float(y)
        out = X, y
        
    return out


from mlquantify._config import get_config


from scipy.special import softmax

def validate_prevalences(
    quantifier,
    prevalences: np.ndarray | list | dict,
    classes: np.ndarray,
    return_type: str | None = None,
    normalize: bool | None = True,
    normalization: str | None = None
) -> dict | np.ndarray:
    """Normalize a prevalence estimate and format it according to global config.

    Converts a prevalence array, list, or dict into the configured output
    format (``'dict'`` or array), applying the configured normalization.

    Parameters
    ----------
    quantifier : BaseQuantifier
        The quantifier whose output is being validated (used to read
        global config defaults).
    prevalences : array-like of shape (n_classes,) or dict
        Raw prevalence estimates.  Dicts are keyed by class label.
    classes : array-like of shape (n_classes,)
        Class labels in the desired output order.
    return_type : {'dict', 'array'} or None, default=None
        Output format.  If ``None``, the global
        ``prevalence_return_type`` config setting is used.
    normalize : bool or None, default=True
        Whether to normalize the output.  If ``None``, the global
        ``prevalence_normalization`` config is used.
    normalization : {'sum', 'l1', 'softmax', 'mean', 'median'} or None, \
        default=None
        Explicit normalization method.  Overrides ``normalize`` when set.

    Returns
    -------
    prevalences : dict or ndarray of shape (n_classes,)
        Normalized prevalence estimates in the requested format.
    """
    conf = get_config()
    return_type = return_type or conf["prevalence_return_type"]
    
    if normalization is None:
        normalization = ('sum' if normalize else None) if normalize is not None else conf["prevalence_normalization"]

    if isinstance(prevalences, dict):
        prevalences_arr = np.array([prevalences.get(cls, 0.0) for cls in classes], dtype=float)
    else:
        prevalences_arr = np.asanyarray(prevalences, dtype=float)
    
    result_arr = normalize_prevalences(
        prevalences_arr, 
        classes=classes, 
        method=normalization
    )

    if return_type == "dict":
        np.nan_to_num(result_arr, copy=False, nan=0.0)
        return dict(zip(np.asanyarray(classes).tolist(), result_arr.tolist()))
    
    return result_arr

def normalize_prevalences(
    prevalences: np.ndarray, 
    classes: np.ndarray, 
    method: str | None = 'sum'
) -> np.ndarray:
    """
    Processa a normalização e agregação focando estritamente em arrays.
    """
    if type(prevalences) == dict:
        prevalences = np.array([prevalences.get(cls, 0.0) for cls in classes], dtype=float)
    arr = np.copy(prevalences)
    
    n_classes = len(classes)
    if arr.shape[-1] < n_classes:
        pad_width = [(0, 0)] * arr.ndim
        pad_width[-1] = (0, n_classes - arr.shape[-1])
        arr = np.pad(arr, pad_width, mode='constant')
    elif arr.shape[-1] > n_classes:
        raise ValueError(f"Dimensão de prevalências ({arr.shape[-1]}) maior que o número de classes ({n_classes}).")

    if method in ('sum', 'l1'):
        if arr.ndim == 2:
            row_sums = arr.sum(axis=1, keepdims=True)
            np.divide(arr, row_sums, out=arr, where=row_sums != 0)
            arr = arr.mean(axis=0)

        total = arr.sum()
        if total > 0:
            arr /= total
        else:
            # Nothing to normalize: every entry came out zero. This happens in
            # the one-vs-rest path, where each binary quantifier clips its own
            # estimate to [0, 1] before the estimates are combined -- in a
            # small sample of a hard multiclass problem the adjustment can
            # land below zero for every class at once, and all of them clip to
            # zero. Leaving the vector as it is returns all-zeros as if it were
            # an estimate: it sums to 0, so it is not a prevalence vector at
            # all, and every error measure then reads it as a confident wrong
            # answer rather than as no answer. The uniform distribution is
            # already this library's answer for a non-positive sum: it is what
            # utils.prevalence.normalize_prevalence returns, asserted by
            # test_normalize_prevalence_zero_sum_is_uniform, and what the
            # matching module's normalize_simplex returns. This function was
            # the one place that did not honour it -- and the one the
            # quantifiers go through.
            arr = np.ones_like(arr) / arr.shape[-1]

    elif method == 'softmax':
        from scipy.special import softmax
        if arr.ndim == 2:
            arr = softmax(arr, axis=1).mean(axis=0)
        else:
            arr = softmax(arr)

    elif arr.ndim == 2 and method in ('mean', 'median'):
        arr = np.mean(arr, axis=0) if method == 'mean' else np.median(arr, axis=0)

    return arr

        
        
def check_has_method(obj: Any, method_name: str) -> bool:
    """Check if the object has a callable method with the given name."""
    return callable(getattr(obj, method_name, None))

def check_classes_attribute(quantifier: Any, classes) -> bool:
    """Check if the quantifier has a 'classes_' attribute and if it matches the type of classes."""

    if not hasattr(quantifier, "classes_"):
        return classes

    quantifier_classes = quantifier.classes_

    # Check if types match
    if type(quantifier_classes) != type(classes):
        return classes

    # Check if shapes match before comparing elements
    if len(quantifier_classes) != len(classes) or not np.all(quantifier_classes == classes):
        return classes
    return quantifier_classes


def resolve_aggregate_classes(quantifier, classes=None, *fallbacks):
    """Resolve the ordered class set an ``aggregate`` call should report.

    The output prevalence vector always spans these classes, so a class that is
    absent from the predictions still appears (with prevalence 0).

    The first non-empty candidate is used, in order: the explicit ``classes``
    argument, then each value in ``fallbacks`` (e.g. the training labels, the
    fitted ``classes_``, or the test predictions). The chosen candidate is
    reduced to its sorted unique values and reconciled with any existing
    ``classes_`` via :func:`check_classes_attribute`.

    Parameters
    ----------
    quantifier : object
        The quantifier whose ``classes_`` is being resolved.
    classes : array-like or None, default=None
        Explicit class set requested by the caller. Takes precedence.
    *fallbacks : array-like or None
        Ordered fallbacks used when ``classes`` is ``None``.

    Returns
    -------
    np.ndarray
        The resolved, sorted unique class labels.
    """
    for candidate in (classes, *fallbacks):
        if candidate is None:
            continue
        candidate = np.asarray(candidate)
        if candidate.size == 0:
            continue
        return check_classes_attribute(quantifier, np.unique(candidate))
    raise InvalidParameterError(
        f"{quantifier.__class__.__name__}.aggregate could not determine the "
        "class set: pass `classes=` or fit the quantifier first."
    )

