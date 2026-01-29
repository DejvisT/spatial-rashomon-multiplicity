from typing import Dict, Tuple
import numpy as np


# ---------------------------------------------------------------------
# Core variance-based metric
# ---------------------------------------------------------------------

def prediction_variance(P: np.ndarray) -> np.ndarray:
    """
    Observation-wise variance of predicted probabilities.

    Parameters
    ----------
    P : array of shape (n_models, n_obs)

    Returns
    -------
    v : array of shape (n_obs,)
        Var_m(p_hat_mi)
    """
    return np.var(P, axis=0, ddof=0)


# ---------------------------------------------------------------------
# Normalized variance (optional robustness metric)
# ---------------------------------------------------------------------

def normalized_prediction_variance(
    P: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Variance normalized by Bernoulli variance p(1-p).

    This answers whether variance is large relative to what is possible
    given the mean prediction level.
    """
    mean_p = np.mean(P, axis=0)
    var_p = np.var(P, axis=0, ddof=0)

    denom = np.maximum(mean_p * (1.0 - mean_p), eps)
    return var_p / denom


# ---------------------------------------------------------------------
# Flip-based instability (decision-relevant)
# ---------------------------------------------------------------------

def flip_instability(
    P: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Instability based on how often models cross a decision threshold.

    instability_i = 4 q_i (1 - q_i),
    where q_i = P_m(p_hat_mi > threshold)
    """
    q = np.mean(P > threshold, axis=0)
    return 4.0 * q * (1.0 - q)


# ---------------------------------------------------------------------
# Prediction interval widths
# ---------------------------------------------------------------------

def prediction_interval_width(
    P: np.ndarray,
    lower: float = 0.05,
    upper: float = 0.95,
) -> np.ndarray:
    """
    Width of prediction interval across Rashomon models.

    Useful for diagnostics and plots.
    """
    lo = np.quantile(P, lower, axis=0)
    hi = np.quantile(P, upper, axis=0)
    return hi - lo


# ---------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------

def compute_multiplicity_metrics(
    P: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Compute all supported observation-wise multiplicity metrics.

    Returns
    -------
    dict with keys:
        - 'variance'
        - 'normalized_variance'
        - 'flip_instability'
        - 'pi_width_90'
        - 'pi_width_95'
    """
    return {
        "variance": prediction_variance(P),
        "normalized_variance": normalized_prediction_variance(P),
        "flip_instability": flip_instability(P, threshold=threshold),
        "pi_width_90": prediction_interval_width(P, 0.05, 0.95),
        "pi_width_95": prediction_interval_width(P, 0.025, 0.975),
    }
