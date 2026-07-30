from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TelemetryNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(24, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


@dataclass
class NeuralNetBundle:
    model: TelemetryNet
    scaler: StandardScaler
    device: torch.device
    validation_loss_history: list[float]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_neural_net(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    random_seed: int,
    weighted: bool,
    max_epochs: int = 30,
    patience: int = 5,
) -> NeuralNetBundle:
    _set_seed(random_seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_validation_scaled = scaler.transform(
        x_validation
    ).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(x_train_scaled),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
    )

    validation_features = torch.from_numpy(
        x_validation_scaled
    ).to(device)
    validation_targets = torch.from_numpy(
        y_validation.astype(np.float32)
    ).to(device)

    model = TelemetryNet(input_dim=x_train.shape[1]).to(device)

    positive_count = float(y_train.sum())
    negative_count = float(len(y_train) - positive_count)
    if weighted and positive_count > 0:
        pos_weight = torch.tensor(
            [negative_count / positive_count],
            dtype=torch.float32,
            device=device,
        )
    else:
        pos_weight = torch.tensor(
            [1.0],
            dtype=torch.float32,
            device=device,
        )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.2e-3,
        weight_decay=1e-4,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    validation_loss_history: list[float] = []

    for _epoch in range(max_epochs):
        model.train()

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            validation_logits = model(validation_features)
            validation_loss = float(
                criterion(
                    validation_logits,
                    validation_targets,
                ).item()
            )

        validation_loss_history.append(validation_loss)

        if validation_loss < best_validation_loss - 1e-4:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()

    return NeuralNetBundle(
        model=model,
        scaler=scaler,
        device=device,
        validation_loss_history=validation_loss_history,
    )


def predict_neural_net_probability(
    bundle: NeuralNetBundle,
    features: np.ndarray,
) -> np.ndarray:
    scaled = bundle.scaler.transform(features).astype(np.float32)
    tensor = torch.from_numpy(scaled).to(bundle.device)

    bundle.model.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(bundle.model(tensor))

    return probabilities.detach().cpu().numpy()


def save_neural_net(
    bundle: NeuralNetBundle,
    model_path: Path,
    scaler_path: Path,
    input_dim: int,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_dim": input_dim,
            "state_dict": bundle.model.state_dict(),
            "validation_loss_history": (
                bundle.validation_loss_history
            ),
        },
        model_path,
    )
    joblib.dump(bundle.scaler, scaler_path)
