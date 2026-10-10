import numpy as np
import pandas as pd

def calibrate_scores_by_train_ecdf(train_scores, eval_scores) -> np.ndarray:
    """Map scores to percentiles of the fitted model's training-score distribution."""
    reference = np.sort(np.asarray(train_scores, dtype=float))
    if reference.size == 0:
        raise ValueError("train_scores must not be empty")
    values = np.asarray(eval_scores, dtype=float)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return (left + right) / (2.0 * reference.size)

def add_foundational_feature(df: pd.DataFrame) -> pd.DataFrame:
    text = (
        df["feature_name"].fillna("").str.lower()
        + " "
        + df["test_scenario"].fillna("").str.lower()
    )
    foundation_terms = [
        "login",
        "access",
        "auth",
        "authentication",
        "execute a test cycle",
        "core business function",
        "server",
        "cli",
        "system access",
        "inaccessible",
    ]
    df["is_foundational"] = text.apply(lambda x: int(any(term in x for term in foundation_terms)))
    return df

def fit_bradley_terry_map(n: int, wins: np.ndarray, prior_mu: np.ndarray,
                          lam: float = 1.0, lr: float = 0.1,
                          n_iter: int = 5000, tol: float = 1e-8) -> np.ndarray:
    """Fit mean-centered log strengths with a Gaussian prior."""
    games = wins + wins.T
    wins_i = wins.sum(axis=1)
    s = prior_mu.copy().astype(float)
    for _ in range(n_iter):
        diff = s[:, None] - s[None, :]
        p = 1.0 / (1.0 + np.exp(-diff))      # p[i,j] = P(i beats j)
        exp_wins = (games * p).sum(axis=1)
        grad = (wins_i - exp_wins) - lam * (s - prior_mu)
        s_new = s + lr * grad
        s_new -= s_new.mean()
        if np.max(np.abs(s_new - s)) < tol:
            s = s_new
            break
        s = s_new
    return s
