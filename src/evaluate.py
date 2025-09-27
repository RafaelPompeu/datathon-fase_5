"""Avaliação do modelo treinado."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

from .model_utils import load_pipeline
from .preprocessing import preprocess_dataset
from .feature_engineering import split_features_target
from .utils import DATA_DIR, POSITIVE_STATUSES


def evaluate(
    model_path: Path,
    data_dir: Path = DATA_DIR,
    output_path: Path | None = None,
) -> dict:
    dataset = preprocess_dataset(data_dir=data_dir, positive_statuses=POSITIVE_STATUSES)
    pipeline = load_pipeline(model_path)

    X, y = split_features_target(dataset)
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred).tolist()

    metrics = {
        "classification_report": report,
        "confusion_matrix": cm,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia modelo treinado no dataset completo")
    parser.add_argument("--model-path", type=Path, default=Path("models/modelo_decision.pkl"))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args.model_path, data_dir=args.data_dir, output_path=args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
