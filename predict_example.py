from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from src.config import FEATURE_COLUMNS


def main() -> None:
    project_root = Path(__file__).resolve().parent
    summary_path = project_root / "outputs" / "summary.json"
    model_path = (
        project_root
        / "outputs"
        / "models"
        / "xgboost_weighted.json"
    )

    if not summary_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Run 'python run_project.py' before inference."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    threshold = float(
        summary["thresholds"]["XGBoost Weighted"]
    )

    model = XGBClassifier()
    model.load_model(model_path)

    sample = {
        "operating_load": 0.88,
        "ambient_temperature": 31.0,
        "temperature": 96.0,
        "vibration": 8.2,
        "pressure": 5.1,
        "rpm": 3_250.0,
        "motor_current": 58.0,
        "acoustic_level": 82.0,
        "flow_rate": 88.0,
        "hours_since_maintenance": 1_760.0,
    }

    features = np.array(
        [[sample[column] for column in FEATURE_COLUMNS]],
        dtype=np.float32,
    )
    probability = float(model.predict_proba(features)[0, 1])
    prediction = int(probability >= threshold)

    print("Telemetry sample:")
    print(json.dumps(sample, indent=2))
    print()
    print(f"Anomaly probability: {probability:.4f}")
    print(f"Safety threshold: {threshold:.4f}")
    print("Prediction:", "ANOMALY" if prediction else "NORMAL")


if __name__ == "__main__":
    main()
