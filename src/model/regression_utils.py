"""Shared OLS/ridge fitting + cross-validation — used by train_fertility.py and
train_survival.py so this logic exists exactly once (skills.md §3 spirit).

Small-sample discipline throughout (SKILL.md §7): linear only, never a tree ensemble.
"""
import numpy as np


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(X)), X])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coeffs


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """L2-regularized fit in standardized feature space, converted back to raw-feature-scale
    coefficients so callers don't need to know a ridge fit was used."""
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0  # a constant column shouldn't divide by zero

    Xs = (X - means) / stds
    n, p = Xs.shape
    design = np.column_stack([np.ones(n), Xs])
    penalty = alpha * np.eye(p + 1)
    penalty[0, 0] = 0.0  # never penalize the intercept
    coeffs_std = np.linalg.solve(design.T @ design + penalty, design.T @ y)

    raw_coeffs = coeffs_std[1:] / stds
    raw_intercept = coeffs_std[0] - np.sum(raw_coeffs * means)
    return np.concatenate([[raw_intercept], raw_coeffs])


def predict(coeffs: np.ndarray, X: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(X)), X])
    return design @ coeffs


def _fit_metrics(y: np.ndarray, preds: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(preds - y)))
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"r2": r2, "mae": mae, "n": len(y)}


def loocv(X: np.ndarray, y: np.ndarray, fit_fn) -> dict:
    """Leave-one-*row*-out CV. Only valid when rows are independent — for panel data with
    repeated per-group (e.g. per-district) rows, use leave_one_group_out_cv instead, since rows
    from the same group share unobserved local factors that row-level LOOCV would leak across
    train/test.
    """
    n = len(X)
    preds = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        coeffs = fit_fn(X[mask], y[mask])
        preds[i] = predict(coeffs, X[i:i + 1])[0]
    return _fit_metrics(y, preds)


def leave_one_group_out_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, fit_fn) -> dict:
    """Leave-one-*district*-out CV: for each unique group, fit on every row NOT in that group,
    predict every row that IS in that group. This is the statistically correct method for panel
    data pooled across districts — a naive row-level LOOCV would let other years of the same
    district leak into training when predicting a held-out year from that district, overstating
    how well the model actually generalizes to a district it hasn't seen.
    """
    preds = np.empty(len(X))
    for group in np.unique(groups):
        test_mask = groups == group
        train_mask = ~test_mask
        coeffs = fit_fn(X[train_mask], y[train_mask])
        preds[test_mask] = predict(coeffs, X[test_mask])
    return _fit_metrics(y, preds)
