from src.data_generation import generate_synthetic_telemetry
from src.drift import drift_summary, simulate_covariate_drift


def test_drift_simulation_changes_distribution():
    frame = generate_synthetic_telemetry(
        n_samples=2_000,
        n_machines=4,
        seed=9,
    )
    drifted = simulate_covariate_drift(frame, seed=10)
    summary = drift_summary(frame, drifted)

    temperature_psi = float(
        summary.loc[
            summary["feature"] == "temperature",
            "psi",
        ].iloc[0]
    )

    assert drifted["temperature"].mean() > frame["temperature"].mean()
    assert temperature_psi > 0
