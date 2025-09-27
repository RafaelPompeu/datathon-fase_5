"""Transformações e engenharia de variáveis."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


TEXT_COLUMN = "combined_text"
CATEGORICAL_COLUMNS = [
    "tipo_contratacao",
    "prioridade_vaga",
    "nivel_profissional_cand",
    "nivel_ingles_cand",
]


def build_feature_transformer(max_features: int = 5000) -> ColumnTransformer:
    """Retorna transformador com TF-IDF + one-hot."""
    text_transformer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    transformer = ColumnTransformer(
        transformers=[
            ("text", text_transformer, TEXT_COLUMN),
            ("categorical", categorical_transformer, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    return transformer


def build_training_pipeline(max_features: int = 5000, **model_kwargs) -> Pipeline:
    """Cria pipeline com transformações e modelo de classificação."""
    transformer = build_feature_transformer(max_features=max_features)
    classifier = LogisticRegression(max_iter=model_kwargs.pop("max_iter", 1000), **model_kwargs)
    pipeline = Pipeline(
        steps=[
            ("features", transformer),
            ("classifier", classifier),
        ]
    )
    return pipeline


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa variáveis independentes e rótulo."""
    X = df[[TEXT_COLUMN] + CATEGORICAL_COLUMNS].copy()
    y = df["label"].astype(int)
    return X, y
