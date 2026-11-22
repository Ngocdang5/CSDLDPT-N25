"""scaling.py - Chuan hoa Z-score. Tham so chi tinh tren tap database, dung lai cho truy van."""
import numpy as np

EPS = 1e-12


def fit_scaler(X):
    """X: (n, d) -> (mean, std). Chieu hang so (std ~ 0) dat std = 1 de khong chia cho 0."""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)                    # std tong the (ddof = 0)
    constant = std < EPS
    std = np.where(constant, 1.0, std)
    return mean, std, constant


def transform(X, mean, std):
    return (np.asarray(X, dtype=np.float64) - mean) / std
