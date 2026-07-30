from __future__ import annotations

from pathlib import Path

import numpy as np
from xgboost import XGBClassifier


def build_xgboost(
    random_seed: int,
    scale_pos_weight: float = 1.0,
) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=260,
        max_depth=5,
        learning_rate=0.045,
        min_child_weight=2.0,
        subsample=0.88,
        colsample_bytree=0.88,
        reg_alpha=0.05,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed,
        n_jobs=-1,
        tree_method="hist",
    )


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    random_seed: int,
    weighted: bool,
) -> XGBClassifier:
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    scale_pos_weight = (
        negative_count / positive_count
        if weighted and positive_count > 0
        else 1.0
    )

    model = build_xgboost(
        random_seed=random_seed,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        verbose=False,
    )
    return model


def predict_xgboost_probability(
    model: XGBClassifier,
    features: np.ndarray,
) -> np.ndarray:
    return model.predict_proba(features)[:, 1]


def save_xgboost(model: XGBClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
