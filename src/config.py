from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "operating_load",
    "ambient_temperature",
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "motor_current",
    "acoustic_level",
    "flow_rate",
    "hours_since_maintenance",
]

TARGET_COLUMN = "anomaly"

@dataclass(frozen=True)
class RunConfig:
    project_root: Path
    n_samples: int = 10_000
    n_machines: int = 12
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    random_seed: int = RANDOM_SEED
    neural_net_epochs: int = 30
    neural_net_patience: int = 5
    minimum_precision_for_safety_threshold: float = 0.40

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def eda_dir(self) -> Path:
        return self.outputs_dir / "eda"

    @property
    def metrics_dir(self) -> Path:
        return self.outputs_dir / "metrics"

    @property
    def models_dir(self) -> Path:
        return self.outputs_dir / "models"

    def create_directories(self) -> None:
        for directory in [
            self.data_dir,
            self.outputs_dir,
            self.eda_dir,
            self.metrics_dir,
            self.models_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
