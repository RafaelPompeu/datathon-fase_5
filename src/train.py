"""Pipeline de treinamento do modelo de matching."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .feature_engineering import build_training_pipeline, split_features_target
from .preprocessing import preprocess_dataset
from .utils import (
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    POSITIVE_STATUSES,
    ensure_directories,
    setup_logging,
)
from .model_utils import save_pipeline


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(set(y_true)) > 1 else 0.0,
    }


def train(
    data_dir: Path = DATA_DIR,
    model_path: Path = MODELS_DIR / "modelo_decision.pkl",
    metrics_path: Path = MODELS_DIR / "metrics.json",
    test_size: float = 0.3,
    random_state: int = 42,
    max_features: int = 5000,
) -> Dict[str, float]:
    setup_logging()
    ensure_directories([model_path.parent, metrics_path.parent, REPORTS_DIR])

    dataset = preprocess_dataset(data_dir=data_dir, positive_statuses=POSITIVE_STATUSES)
    X, y = split_features_target(dataset)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = build_training_pipeline(max_features=max_features)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_pred, y_proba)

    save_pipeline(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    evaluation_df = pd.DataFrame({
        "y_true": y_val,
        "y_pred": y_pred,
        "y_proba": y_proba,
    })
    evaluation_df.to_csv(REPORTS_DIR / "validation_predictions.csv", index=False)

    dataset.to_csv(REPORTS_DIR / "dataset_preparado.csv", index=False)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o modelo de matching Decision")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Diretório com os JSON de dados")
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "modelo_decision.pkl")
    parser.add_argument("--metrics-path", type=Path, default=MODELS_DIR / "metrics.json")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train(
        data_dir=args.data_dir,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
    )
    print("Métricas de validação:")
    for metric, value in metrics.items():
        print(f"- {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
