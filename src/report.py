from __future__ import annotations

from pathlib import Path

import pandas as pd


def select_safety_candidate(
    clean_metrics: pd.DataFrame,
    drift_metrics: pd.DataFrame,
) -> str:
    clean = clean_metrics.set_index("model")
    drift = drift_metrics.set_index("model")

    candidates = []
    for model_name in clean.index:
        worst_case_recall = min(
            float(clean.loc[model_name, "recall"]),
            float(drift.loc[model_name, "recall"]),
        )
        worst_case_fnr = max(
            float(clean.loc[model_name, "false_negative_rate"]),
            float(drift.loc[model_name, "false_negative_rate"]),
        )
        precision = float(clean.loc[model_name, "precision"])
        latency = float(
            clean.loc[model_name, "inference_ms_per_sample"]
        )
        candidates.append(
            (
                model_name,
                worst_case_recall,
                -worst_case_fnr,
                precision,
                -latency,
            )
        )

    return max(candidates, key=lambda row: row[1:])[0]


def write_markdown_report(
    output_path: Path,
    dataset_size: int,
    anomaly_rate: float,
    clean_metrics: pd.DataFrame,
    drift_metrics: pd.DataFrame,
    drift_table: pd.DataFrame,
) -> str:
    selected_model = select_safety_candidate(
        clean_metrics,
        drift_metrics,
    )

    clean_view = clean_metrics[
        [
            "model",
            "precision",
            "recall",
            "f1",
            "false_negative_rate",
            "roc_auc",
            "average_precision",
            "inference_ms_per_sample",
            "threshold",
        ]
    ].copy()
    drift_view = drift_metrics[
        [
            "model",
            "precision",
            "recall",
            "f1",
            "false_negative_rate",
            "inference_ms_per_sample",
        ]
    ].copy()

    markdown = f"""# Predictive Maintenance Anomaly Detection Report

## Dataset

- Synthetic telemetry observations: **{dataset_size:,}**
- Anomaly prevalence: **{anomaly_rate:.2%}**
- Split strategy: chronological train / validation / test
- Sensors: temperature, vibration, pressure, RPM, motor current,
  acoustic level, flow rate, load and maintenance age

## Clean Test Performance

{clean_view.to_markdown(index=False, floatfmt=".4f")}

## Drifted Test Performance

{drift_view.to_markdown(index=False, floatfmt=".4f")}

## Drift Diagnostics

{drift_table.head(10).to_markdown(index=False, floatfmt=".4f")}

## Safety-Critical Model Decision

The automatic candidate selected by the project's conservative ranking is:

**{selected_model}**

The ranking prioritizes the worst-case recall across clean and drifted
test sets, then the worst-case false-negative rate, clean precision and
inference latency.

For a safety-critical system, a missed anomaly may be more costly than
a false alarm. Therefore recall and false-negative rate receive priority.
Class weighting is tested for both model families. A separate CSV also
reports all models at the default 0.50 threshold, while the main comparison
uses a validation-only safety threshold to reduce missed anomalies.

However, this project uses synthetic data. No model in this experiment
should be deployed as the sole safety barrier. A real deployment would
also require real failure data, out-of-time validation, sensor quality
checks, drift monitoring, uncertainty handling, engineering redundancy,
alarm escalation rules and human review.

## Interpretation Guide

- **Precision:** Of the alarms raised, how many were real anomalies?
- **Recall:** Of all real anomalies, how many were detected?
- **F1:** Balance between precision and recall.
- **False-negative rate:** Fraction of real anomalies that were missed.
- **Inference time:** Approximate prediction latency per observation.
- **PSI:** Distribution-shift indicator, not proof of causal degradation.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return selected_model
