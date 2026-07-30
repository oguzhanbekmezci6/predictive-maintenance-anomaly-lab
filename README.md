# Predictive Maintenance Anomaly Detection Lab

An end-to-end, safety-oriented machine-learning project using synthetic
industrial telemetry.

## Highlights

- Reproducible synthetic sensor telemetry
- Rare anomaly labels and class-imbalance analysis
- XGBoost and PyTorch neural-network baselines
- Class-weighted variants
- Validation-only safety threshold selection
- Precision, recall, F1, false-negative rate and inference latency
- Covariate-drift simulation
- Population Stability Index
- Automatic Markdown, CSV, JSON and PNG reports

## Run

Windows:

```text
START_PROJECT.bat
```

Or:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python run_project.py
```

Quick run:

```bash
python run_project.py --samples 5000 --epochs 15
```

The main report is written to:

```text
outputs/REPORT.md
```

## Important limitation

The project uses synthetic telemetry and is intended for learning and
portfolio demonstration. It must not be used as the sole safety barrier in
a real industrial system.
