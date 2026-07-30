from __future__ import annotations

import argparse
from pathlib import Path

from src.config import RunConfig
from src.pipeline import run_pipeline


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic predictive-maintenance anomaly detection lab"
        )
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10_000,
        help="Number of synthetic telemetry rows.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum neural-network epochs.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parent

    config = RunConfig(
        project_root=project_root,
        n_samples=arguments.samples,
        neural_net_epochs=arguments.epochs,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
