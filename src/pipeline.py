from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, RunConfig
from src.data_generation import (
    chronological_split,
    generate_synthetic_telemetry,
)
from src.drift import drift_summary, simulate_covariate_drift
from src.eda import run_eda
from src.metrics import (
    binary_classification_metrics,
    choose_safety_threshold,
    confusion_matrix_frame,
    measure_inference_ms_per_sample,
)
from src.models.neural_net import (
    NeuralNetBundle,
    predict_neural_net_probability,
    save_neural_net,
    train_neural_net,
)
from src.models.xgboost_model import (
    predict_xgboost_probability,
    save_xgboost,
    train_xgboost,
)
from src.report import write_markdown_report


def _to_arrays(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    x = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = frame[TARGET_COLUMN].to_numpy(dtype=np.int64)
    return x, y


def _evaluate_model(
    model_name: str,
    probability_function,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    minimum_precision: float,
) -> tuple[dict[str, float | str], np.ndarray]:
    validation_probabilities = probability_function(x_validation)
    threshold = choose_safety_threshold(
        y_validation,
        validation_probabilities,
        minimum_precision=minimum_precision,
    )

    test_probabilities = probability_function(x_test)
    metrics = binary_classification_metrics(
        y_test,
        test_probabilities,
        threshold,
    )
    metrics["model"] = model_name
    metrics["inference_ms_per_sample"] = (
        measure_inference_ms_per_sample(
            probability_function,
            x_test,
        )
    )

    return metrics, test_probabilities


def run_pipeline(config: RunConfig) -> dict:
    config.create_directories()

    print("[1/8] Generating synthetic telemetry...")
    telemetry = generate_synthetic_telemetry(
        n_samples=config.n_samples,
        n_machines=config.n_machines,
        seed=config.random_seed,
    )
    telemetry.to_csv(
        config.data_dir / "telemetry.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train_frame, validation_frame, test_frame = (
        chronological_split(
            telemetry,
            train_ratio=config.train_ratio,
            validation_ratio=config.validation_ratio,
        )
    )

    print("[2/8] Running exploratory data analysis...")
    run_eda(train_frame, config.eda_dir)

    x_train, y_train = _to_arrays(train_frame)
    x_validation, y_validation = _to_arrays(validation_frame)
    x_test, y_test = _to_arrays(test_frame)

    clean_metrics_records: list[dict] = []
    default_threshold_records: list[dict] = []
    clean_probabilities: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    model_probability_functions = {}

    print("[3/8] Training XGBoost baseline and weighted models...")
    for weighted in [False, True]:
        model_name = (
            "XGBoost Weighted"
            if weighted
            else "XGBoost Baseline"
        )
        model = train_xgboost(
            x_train,
            y_train,
            x_validation,
            y_validation,
            random_seed=config.random_seed,
            weighted=weighted,
        )
        probability_function = (
            lambda features, fitted_model=model:
            predict_xgboost_probability(
                fitted_model,
                features,
            )
        )
        metrics, probabilities = _evaluate_model(
            model_name,
            probability_function,
            x_validation,
            y_validation,
            x_test,
            y_test,
            config.minimum_precision_for_safety_threshold,
        )
        clean_metrics_records.append(metrics)

        default_metrics = binary_classification_metrics(
            y_test,
            probabilities,
            threshold=0.50,
        )
        default_metrics["model"] = model_name
        default_threshold_records.append(default_metrics)

        clean_probabilities[model_name] = probabilities
        thresholds[model_name] = float(metrics["threshold"])
        model_probability_functions[model_name] = probability_function

        filename = (
            "xgboost_weighted.json"
            if weighted
            else "xgboost_baseline.json"
        )
        save_xgboost(model, config.models_dir / filename)

    print("[4/8] Training neural-network baseline and weighted models...")
    neural_bundles: dict[str, NeuralNetBundle] = {}

    for weighted in [False, True]:
        model_name = (
            "Neural Net Weighted"
            if weighted
            else "Neural Net Baseline"
        )
        bundle = train_neural_net(
            x_train,
            y_train,
            x_validation,
            y_validation,
            random_seed=config.random_seed,
            weighted=weighted,
            max_epochs=config.neural_net_epochs,
            patience=config.neural_net_patience,
        )
        neural_bundles[model_name] = bundle
        probability_function = (
            lambda features, fitted_bundle=bundle:
            predict_neural_net_probability(
                fitted_bundle,
                features,
            )
        )
        metrics, probabilities = _evaluate_model(
            model_name,
            probability_function,
            x_validation,
            y_validation,
            x_test,
            y_test,
            config.minimum_precision_for_safety_threshold,
        )
        clean_metrics_records.append(metrics)

        default_metrics = binary_classification_metrics(
            y_test,
            probabilities,
            threshold=0.50,
        )
        default_metrics["model"] = model_name
        default_threshold_records.append(default_metrics)

        clean_probabilities[model_name] = probabilities
        thresholds[model_name] = float(metrics["threshold"])
        model_probability_functions[model_name] = probability_function

        suffix = "weighted" if weighted else "baseline"
        save_neural_net(
            bundle,
            config.models_dir / f"neural_net_{suffix}.pt",
            config.models_dir / f"scaler_{suffix}.joblib",
            input_dim=len(FEATURE_COLUMNS),
        )

    clean_metrics = pd.DataFrame(clean_metrics_records).sort_values(
        ["false_negative_rate", "f1"],
        ascending=[True, False],
    )
    clean_metrics.to_csv(
        config.metrics_dir / "model_comparison_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    default_threshold_metrics = pd.DataFrame(
        default_threshold_records
    ).sort_values(
        ["false_negative_rate", "f1"],
        ascending=[True, False],
    )
    default_threshold_metrics.to_csv(
        config.metrics_dir
        / "model_comparison_default_threshold_0_50.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("[5/8] Saving confusion matrices...")
    for model_name, probabilities in clean_probabilities.items():
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        confusion_matrix_frame(
            y_test,
            probabilities,
            thresholds[model_name],
        ).to_csv(
            config.metrics_dir
            / f"confusion_matrix_{safe_name}.csv",
            encoding="utf-8-sig",
        )

    print("[6/8] Simulating drift and re-evaluating...")
    drifted_test = simulate_covariate_drift(
        test_frame,
        seed=config.random_seed + 1,
    )
    drifted_test.to_csv(
        config.data_dir / "telemetry_drifted_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    x_drift, y_drift = _to_arrays(drifted_test)

    drift_table = drift_summary(train_frame, drifted_test)
    drift_table.to_csv(
        config.metrics_dir / "feature_drift_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    drift_metrics_records: list[dict] = []
    for model_name, probability_function in (
        model_probability_functions.items()
    ):
        probabilities = probability_function(x_drift)
        metrics = binary_classification_metrics(
            y_drift,
            probabilities,
            thresholds[model_name],
        )
        metrics["model"] = model_name
        metrics["inference_ms_per_sample"] = (
            measure_inference_ms_per_sample(
                probability_function,
                x_drift,
            )
        )
        drift_metrics_records.append(metrics)

    drift_metrics = pd.DataFrame(
        drift_metrics_records
    ).sort_values(
        ["false_negative_rate", "f1"],
        ascending=[True, False],
    )
    drift_metrics.to_csv(
        config.metrics_dir / "model_comparison_drift.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("[7/8] Writing safety-oriented report...")
    selected_model = write_markdown_report(
        output_path=config.outputs_dir / "REPORT.md",
        dataset_size=len(telemetry),
        anomaly_rate=float(telemetry[TARGET_COLUMN].mean()),
        clean_metrics=clean_metrics,
        drift_metrics=drift_metrics,
        drift_table=drift_table,
    )

    summary = {
        "dataset": {
            "rows": len(telemetry),
            "train_rows": len(train_frame),
            "validation_rows": len(validation_frame),
            "test_rows": len(test_frame),
            "anomaly_rate": float(
                telemetry[TARGET_COLUMN].mean()
            ),
            "features": FEATURE_COLUMNS,
        },
        "selected_safety_candidate": selected_model,
        "thresholds": thresholds,
        "clean_metrics": clean_metrics.to_dict(
            orient="records"
        ),
        "drift_metrics": drift_metrics.to_dict(
            orient="records"
        ),
    }
    (config.outputs_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[8/8] Completed.")
    print()
    print("Recommended safety-oriented candidate:", selected_model)
    print(
        clean_metrics[
            [
                "model",
                "precision",
                "recall",
                "f1",
                "false_negative_rate",
                "inference_ms_per_sample",
            ]
        ].to_string(index=False)
    )
    print()
    print("Report:", config.outputs_dir / "REPORT.md")

    return summary
