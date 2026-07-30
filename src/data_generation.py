from __future__ import annotations

import numpy as np
import pandas as pd


ANOMALY_TYPES = [
    "overheating",
    "bearing_fault",
    "pressure_fault",
    "overspeed",
    "combined_fault",
]


def generate_synthetic_telemetry(
    n_samples: int = 10_000,
    n_machines: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Create reproducible, imbalanced synthetic industrial telemetry.

    The target is intentionally rare. Anomalies alter physically related
    sensor groups so that the problem is learnable but not perfectly trivial.
    """
    if n_samples < 1_000:
        raise ValueError("n_samples must be at least 1,000.")
    if n_machines < 2:
        raise ValueError("n_machines must be at least 2.")

    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        start="2025-01-01",
        periods=n_samples,
        freq="5min",
    )
    machine_id = rng.integers(1, n_machines + 1, size=n_samples)

    machine_temperature_bias = rng.normal(0.0, 1.5, size=n_machines + 1)
    machine_vibration_bias = rng.normal(0.0, 0.18, size=n_machines + 1)
    machine_risk = rng.normal(0.0, 0.28, size=n_machines + 1)

    operating_load = rng.beta(4.0, 2.2, size=n_samples)
    ambient_temperature = rng.normal(24.0, 5.0, size=n_samples)

    hours_since_maintenance = rng.uniform(0.0, 2_000.0, size=n_samples)
    maintenance_cycle = (np.arange(n_samples) % 2_400) / 2_400
    hours_since_maintenance = (
        0.65 * hours_since_maintenance
        + 0.35 * maintenance_cycle * 2_000
    )

    temperature = (
        49.0
        + 30.0 * operating_load
        + 0.24 * ambient_temperature
        + machine_temperature_bias[machine_id]
        + rng.normal(0.0, 2.8, size=n_samples)
    )
    vibration = (
        0.75
        + 2.8 * operating_load
        + 0.00055 * hours_since_maintenance
        + machine_vibration_bias[machine_id]
        + rng.normal(0.0, 0.32, size=n_samples)
    )
    pressure = (
        5.7
        + 2.1 * operating_load
        + rng.normal(0.0, 0.28, size=n_samples)
    )
    rpm = (
        1_250
        + 1_650 * operating_load
        + rng.normal(0.0, 85.0, size=n_samples)
    )
    motor_current = (
        13.0
        + 39.0 * operating_load
        + 0.055 * temperature
        + rng.normal(0.0, 2.4, size=n_samples)
    )
    acoustic_level = (
        54.0
        + 13.0 * operating_load
        + 1.05 * vibration
        + rng.normal(0.0, 1.8, size=n_samples)
    )
    flow_rate = (
        118.0
        - 21.0 * operating_load
        + 2.7 * pressure
        + rng.normal(0.0, 3.5, size=n_samples)
    )

    # Rare-event probability rises with load, age and machine-specific risk.
    anomaly_logit = (
        -4.45
        + 1.15 * operating_load
        + 0.00045 * hours_since_maintenance
        + machine_risk[machine_id]
    )
    anomaly_probability = 1.0 / (1.0 + np.exp(-anomaly_logit))
    anomaly = rng.binomial(1, anomaly_probability, size=n_samples).astype(int)

    anomaly_type = np.full(n_samples, "normal", dtype=object)
    anomaly_indices = np.flatnonzero(anomaly == 1)

    if anomaly_indices.size:
        assigned_types = rng.choice(
            ANOMALY_TYPES,
            size=anomaly_indices.size,
            p=[0.24, 0.31, 0.20, 0.15, 0.10],
        )
        anomaly_type[anomaly_indices] = assigned_types

        for fault_type in ANOMALY_TYPES:
            idx = anomaly_indices[assigned_types == fault_type]
            if idx.size == 0:
                continue

            if fault_type == "overheating":
                temperature[idx] += rng.uniform(14.0, 28.0, size=idx.size)
                motor_current[idx] += rng.uniform(3.0, 8.0, size=idx.size)

            elif fault_type == "bearing_fault":
                vibration[idx] += rng.uniform(3.0, 7.5, size=idx.size)
                acoustic_level[idx] += rng.uniform(8.0, 18.0, size=idx.size)

            elif fault_type == "pressure_fault":
                pressure[idx] -= rng.uniform(1.5, 3.1, size=idx.size)
                flow_rate[idx] -= rng.uniform(12.0, 28.0, size=idx.size)

            elif fault_type == "overspeed":
                rpm[idx] += rng.uniform(420.0, 900.0, size=idx.size)
                vibration[idx] += rng.uniform(1.4, 3.2, size=idx.size)

            elif fault_type == "combined_fault":
                temperature[idx] += rng.uniform(10.0, 22.0, size=idx.size)
                vibration[idx] += rng.uniform(2.4, 5.5, size=idx.size)
                pressure[idx] -= rng.uniform(0.9, 2.2, size=idx.size)
                motor_current[idx] += rng.uniform(4.0, 9.0, size=idx.size)

    # Small label noise prevents a perfectly deterministic toy task.
    noise_count = max(1, int(n_samples * 0.003))
    noise_idx = rng.choice(n_samples, size=noise_count, replace=False)
    anomaly[noise_idx] = 1 - anomaly[noise_idx]
    anomaly_type[(anomaly == 0)] = "normal"
    anomaly_type[(anomaly == 1) & (anomaly_type == "normal")] = "unclassified_fault"

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "machine_id": machine_id,
            "operating_load": np.clip(operating_load, 0.0, 1.0),
            "ambient_temperature": ambient_temperature,
            "temperature": np.clip(temperature, 20.0, 130.0),
            "vibration": np.clip(vibration, 0.05, 20.0),
            "pressure": np.clip(pressure, 0.4, 12.0),
            "rpm": np.clip(rpm, 300.0, 4_500.0),
            "motor_current": np.clip(motor_current, 2.0, 100.0),
            "acoustic_level": np.clip(acoustic_level, 25.0, 120.0),
            "flow_rate": np.clip(flow_rate, 10.0, 180.0),
            "hours_since_maintenance": np.clip(
                hours_since_maintenance,
                0.0,
                2_500.0,
            ),
            "anomaly_type": anomaly_type,
            "anomaly": anomaly,
        }
    )

    return frame.sort_values("timestamp").reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("Split ratios must be positive.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train + validation ratio must be below 1.")

    n_rows = len(frame)
    train_end = int(n_rows * train_ratio)
    validation_end = int(n_rows * (train_ratio + validation_ratio))

    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:validation_end].copy()
    test = frame.iloc[validation_end:].copy()

    return train, validation, test
