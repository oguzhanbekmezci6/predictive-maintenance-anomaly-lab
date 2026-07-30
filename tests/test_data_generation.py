from src.config import FEATURE_COLUMNS
from src.data_generation import generate_synthetic_telemetry


def test_generated_data_is_valid():
    frame = generate_synthetic_telemetry(
        n_samples=2_000,
        n_machines=4,
        seed=7,
    )

    assert len(frame) == 2_000
    assert frame[FEATURE_COLUMNS].isna().sum().sum() == 0
    assert set(frame["anomaly"].unique()).issubset({0, 1})

    anomaly_rate = frame["anomaly"].mean()
    assert 0.01 < anomaly_rate < 0.20
