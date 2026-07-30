from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN


def run_eda(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = frame[FEATURE_COLUMNS + [TARGET_COLUMN]].describe().T
    summary.to_csv(output_dir / "feature_summary.csv", encoding="utf-8-sig")

    class_counts = (
        frame[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .rename_axis("anomaly")
        .reset_index(name="count")
    )
    class_counts["percentage"] = class_counts["count"] / len(frame) * 100
    class_counts.to_csv(
        output_dir / "class_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_class = frame.groupby(TARGET_COLUMN)[FEATURE_COLUMNS].mean().T
    by_class.columns = ["normal_mean", "anomaly_mean"]
    by_class["absolute_difference"] = (
        by_class["anomaly_mean"] - by_class["normal_mean"]
    ).abs()
    by_class.to_csv(
        output_dir / "feature_means_by_class.csv",
        encoding="utf-8-sig",
    )

    figure = plt.figure(figsize=(7, 5))
    axes = figure.add_subplot(111)
    axes.bar(
        ["Normal", "Anomaly"],
        [
            int((frame[TARGET_COLUMN] == 0).sum()),
            int((frame[TARGET_COLUMN] == 1).sum()),
        ],
    )
    axes.set_title("Class Distribution")
    axes.set_ylabel("Observation Count")
    figure.tight_layout()
    figure.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close(figure)

    correlation = frame[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
    correlation.to_csv(
        output_dir / "correlation_matrix.csv",
        encoding="utf-8-sig",
    )

    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111)
    image = axes.imshow(correlation.to_numpy(), aspect="auto")
    axes.set_xticks(np.arange(len(correlation.columns)))
    axes.set_yticks(np.arange(len(correlation.index)))
    axes.set_xticklabels(correlation.columns, rotation=75, ha="right")
    axes.set_yticklabels(correlation.index)
    axes.set_title("Correlation Matrix")
    figure.colorbar(image, ax=axes)
    figure.tight_layout()
    figure.savefig(output_dir / "correlation_matrix.png", dpi=160)
    plt.close(figure)

    standardized = (
        frame[FEATURE_COLUMNS] - frame[FEATURE_COLUMNS].mean()
    ) / frame[FEATURE_COLUMNS].std(ddof=0).replace(0, 1)

    normal_values = standardized.loc[
        frame[TARGET_COLUMN] == 0,
        "vibration",
    ].to_numpy()
    anomaly_values = standardized.loc[
        frame[TARGET_COLUMN] == 1,
        "vibration",
    ].to_numpy()

    figure = plt.figure(figsize=(7, 5))
    axes = figure.add_subplot(111)
    axes.boxplot(
        [normal_values, anomaly_values],
        tick_labels=["Normal", "Anomaly"],
    )
    axes.set_title("Standardized Vibration by Class")
    axes.set_ylabel("Standardized Vibration")
    figure.tight_layout()
    figure.savefig(
        output_dir / "vibration_by_class.png",
        dpi=160,
    )
    plt.close(figure)
