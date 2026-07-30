from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS


def simulate_covariate_drift(
    test_frame: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Shift sensor distributions while preserving the original labels."""
    rng = np.random.default_rng(seed)
    drifted = test_frame.copy()

    row_count = len(drifted)

    drifted["ambient_temperature"] += rng.normal(
        4.0,
        1.0,
        size=row_count,
    )
    drifted["operating_load"] = np.clip(
        drifted["operating_load"] + rng.normal(0.08, 0.025, row_count),
        0.0,
        1.0,
    )
    drifted["temperature"] += rng.normal(7.0, 1.8, row_count)
    drifted["vibration"] *= rng.normal(1.22, 0.035, row_count)
    drifted["pressure"] -= rng.normal(0.32, 0.09, row_count)
    drifted["rpm"] *= rng.normal(1.035, 0.008, row_count)
    drifted["motor_current"] *= rng.normal(1.08, 0.018, row_count)
    drifted["acoustic_level"] += rng.normal(2.8, 0.8, row_count)
    drifted["flow_rate"] *= rng.normal(0.94, 0.012, row_count)
    drifted["hours_since_maintenance"] += rng.normal(
        220.0,
        35.0,
        row_count,
    )

    return drifted


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))

    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    reference_ratio = reference_counts / max(reference_counts.sum(), 1)
    current_ratio = current_counts / max(current_counts.sum(), 1)

    epsilon = 1e-6
    reference_ratio = np.clip(reference_ratio, epsilon, None)
    current_ratio = np.clip(current_ratio, epsilon, None)

    psi = np.sum(
        (current_ratio - reference_ratio)
        * np.log(current_ratio / reference_ratio)
    )
    return float(psi)


def drift_summary(
    reference_frame: pd.DataFrame,
    drifted_frame: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []

    for feature in FEATURE_COLUMNS:
        reference = reference_frame[feature].to_numpy()
        current = drifted_frame[feature].to_numpy()
        psi = population_stability_index(reference, current)

        if psi < 0.10:
            severity = "low"
        elif psi < 0.25:
            severity = "moderate"
        else:
            severity = "high"

        reference_mean = float(np.mean(reference))
        current_mean = float(np.mean(current))
        mean_shift_percent = (
            (current_mean - reference_mean)
            / abs(reference_mean)
            * 100
            if reference_mean != 0
            else 0.0
        )

        records.append(
            {
                "feature": feature,
                "reference_mean": reference_mean,
                "drifted_mean": current_mean,
                "mean_shift_percent": mean_shift_percent,
                "psi": psi,
                "severity": severity,
            }
        )

    return pd.DataFrame(records).sort_values(
        "psi",
        ascending=False,
    )
