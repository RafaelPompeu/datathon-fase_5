"""Funções utilitárias para modelos treinados."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from .feature_engineering import split_features_target


def save_pipeline(pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {path}")
    return joblib.load(path)


def predict_probabilities(pipeline, df: pd.DataFrame) -> pd.Series:
    X, _ = split_features_target(df.assign(label=0))
    proba = pipeline.predict_proba(X)[:, 1]
    return pd.Series(proba, index=df.index, name="match_proba")


def rank_candidates_for_job(
    pipeline,
    data: pd.DataFrame,
    job_id: int,
    top_k: int = 5,
) -> pd.DataFrame:
    """Filtra pares vaga-candidato e ranqueia por probabilidade."""
    subset = data[data["job_id"] == job_id].copy()
    if subset.empty:
        raise ValueError(f"Nenhum candidato encontrado para job_id={job_id}")
    scores = predict_probabilities(pipeline, subset)
    subset["match_proba"] = scores
    return subset.sort_values("match_proba", ascending=False).head(top_k)


def add_predictions(pipeline, data: pd.DataFrame) -> pd.DataFrame:
    proba = predict_probabilities(pipeline, data)
    return data.assign(match_proba=proba)
