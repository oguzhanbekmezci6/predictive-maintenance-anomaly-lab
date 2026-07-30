# Predictive Maintenance Anomaly Detection Report

## Dataset

- Synthetic telemetry observations: **10,000**
- Anomaly prevalence: **4.00%**
- Split strategy: chronological train / validation / test
- Sensors: temperature, vibration, pressure, RPM, motor current,
  acoustic level, flow rate, load and maintenance age

## Clean Test Performance

| model               |   precision |   recall |     f1 |   false_negative_rate |   roc_auc |   average_precision |   inference_ms_per_sample |   threshold |
|:--------------------|------------:|---------:|-------:|----------------------:|----------:|--------------------:|--------------------------:|------------:|
| Neural Net Baseline |      0.8615 |   0.9333 | 0.8960 |                0.0667 |    0.9731 |              0.9505 |                    0.4744 |      0.0500 |
| Neural Net Weighted |      0.7568 |   0.9333 | 0.8358 |                0.0667 |    0.9697 |              0.9435 |                    0.2610 |      0.4576 |
| XGBoost Weighted    |      0.9167 |   0.9167 | 0.9167 |                0.0833 |    0.9592 |              0.9364 |                    0.4613 |      0.1743 |
| XGBoost Baseline    |      0.8730 |   0.9167 | 0.8943 |                0.0833 |    0.9588 |              0.9397 |                    0.4943 |      0.0550 |

## Drifted Test Performance

| model               |   precision |   recall |     f1 |   false_negative_rate |   inference_ms_per_sample |
|:--------------------|------------:|---------:|-------:|----------------------:|--------------------------:|
| XGBoost Baseline    |      0.0970 |   0.9667 | 0.1763 |                0.0333 |                    0.6429 |
| Neural Net Baseline |      0.0819 |   0.9667 | 0.1510 |                0.0333 |                    0.5919 |
| Neural Net Weighted |      0.0706 |   0.9667 | 0.1317 |                0.0333 |                    0.6113 |
| XGBoost Weighted    |      0.1234 |   0.9500 | 0.2184 |                0.0500 |                    0.6615 |

## Drift Diagnostics

| feature                 |   reference_mean |   drifted_mean |   mean_shift_percent |    psi | severity   |
|:------------------------|-----------------:|---------------:|---------------------:|-------:|:-----------|
| flow_rate               |         123.2507 |       115.9579 |              -5.9171 | 2.2696 | high       |
| temperature             |          74.4830 |        81.4553 |               9.3609 | 1.0865 | high       |
| vibration               |           3.2080 |         3.9602 |              23.4489 | 0.8654 | high       |
| acoustic_level          |          65.8587 |        68.6656 |               4.2620 | 0.6121 | high       |
| ambient_temperature     |          23.8932 |        27.7242 |              16.0338 | 0.5476 | high       |
| hours_since_maintenance |         995.2212 |      1296.9117 |              30.3139 | 0.4406 | high       |
| pressure                |           7.0311 |         6.7123 |              -4.5341 | 0.4378 | high       |
| operating_load          |           0.6480 |         0.7249 |              11.8677 | 0.2047 | moderate   |
| motor_current           |          42.4386 |        45.7641 |               7.8361 | 0.1805 | moderate   |
| rpm                     |        2321.2738 |      2401.9426 |               3.4752 | 0.0900 | low        |

## Safety-Critical Model Decision

The automatic candidate selected by the project's conservative ranking is:

**Neural Net Baseline**

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
