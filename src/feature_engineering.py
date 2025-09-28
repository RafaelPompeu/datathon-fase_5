"""Transformações e engenharia de variáveis.

Inclui features avançadas:
- Similaridade TF-IDF entre `job_text` e `candidate_text` (cosine)
- Sobreposição de habilidades (competências vs. conhecimentos do candidato)
- Aderência de senioridade (mapeamento ordinal)
"""
from __future__ import annotations

import re
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


TEXT_COLUMNS = ["job_text", "candidate_text"]
CATEGORICAL_COLUMNS = [
    "tipo_contratacao",
    "prioridade_vaga",
    "nivel_ingles_cand",
]

# Colunas necessárias para features numéricas adicionais
NUMERIC_SOURCE_COLUMNS = [
    "job_text",
    "candidate_text",
    "competencias_tecnicas",
    "principais_atividades",
    "conhecimentos_tecnicos",
    "cv_pt",
    "nivel_profissional_req",
    "nivel_profissional_cand",
]

SENIORITY_RANK = {
    "estagiário": 0,
    "trainee": 0,
    "assistente": 0,
    "auxiliar": 0,
    "júnior": 1,
    "junior": 1,
    "analista": 2,
    "pleno": 2,
    "especialista": 3,
    "sênior": 3,
    "senior": 3,
    "líder": 4,
    "lider": 4,
    "supervisor": 4,
    "coordenador": 5,
    "gerente": 6,
    "diretor": 7,
}


class PairwiseTfidfSimilarityAndSkills(BaseEstimator, TransformerMixin):
    """Gera features numéricas a partir de múltiplos campos de texto.

    Saída: matriz (n_samples x 3) com colunas
    [text_similarity, skill_overlap, seniority_score].
    """

    def __init__(self, max_features: int = 8000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            stop_words="portuguese",
        )

    @staticmethod
    def _tokenize_skills(text: str) -> List[str]:
        if text is None:
            return []
        raw = str(text).lower()
        tokens = [t for t in re.split(r"[\s,;\-/\\\n]+", raw) if len(t) >= 3 and t.isalpha()]
        return tokens

    @staticmethod
    def _seniority_rank(value: str) -> int:
        if not isinstance(value, str) or not value:
            return -1
        return SENIORITY_RANK.get(value.strip().lower(), 2)

    @staticmethod
    def _seniority_score(job_value: str, cand_value: str) -> float:
        jr = PairwiseTfidfSimilarityAndSkills._seniority_rank(job_value)
        cr = PairwiseTfidfSimilarityAndSkills._seniority_rank(cand_value)
        if cr < 0 and jr < 0:
            return 0.5
        if cr < 0 or jr < 0:
            return 0.6
        return max(0.0, 1.0 - 0.25 * abs(jr - cr))

    def fit(self, X: pd.DataFrame, y=None):
        texts = list(X["job_text"].fillna("")) + list(X["candidate_text"].fillna(""))
        self.vectorizer.fit(texts)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        job_vecs = self.vectorizer.transform(X["job_text"].fillna(""))
        cand_vecs = self.vectorizer.transform(X["candidate_text"].fillna(""))
        # row-wise cosine similarity
        # Avoid expensive full cosine by using row-wise dot of normalized vectors
        sim_num = (job_vecs.multiply(cand_vecs)).sum(axis=1).A1
        job_norm = np.sqrt(job_vecs.power(2).sum(axis=1)).A1 + 1e-12
        cand_norm = np.sqrt(cand_vecs.power(2).sum(axis=1)).A1 + 1e-12
        text_similarity = sim_num / (job_norm * cand_norm)

        job_skill_text = (
            X.get("competencias_tecnicas", "").fillna("")
            .astype(str)
            .str.cat(X.get("principais_atividades", "").fillna("").astype(str), sep=" ")
            .str.cat(X.get("job_text", "").fillna("").astype(str), sep=" ")
        )
        cand_skill_text = (
            X.get("conhecimentos_tecnicos", "").fillna("")
            .astype(str)
            .str.cat(X.get("cv_pt", "").fillna("").astype(str), sep=" ")
            .str.cat(X.get("candidate_text", "").fillna("").astype(str), sep=" ")
        )

        overlaps = []
        for jtxt, ctxt in zip(job_skill_text.tolist(), cand_skill_text.tolist()):
            jt = set(self._tokenize_skills(jtxt))
            ct = set(self._tokenize_skills(ctxt))
            if not ct:
                overlaps.append(0.0)
            else:
                overlaps.append(len(jt & ct) / max(len(ct), 1))
        overlaps = np.asarray(overlaps, dtype=float)

        seniority_scores = []
        for jr, cr in zip(X.get("nivel_profissional_req", ""), X.get("nivel_profissional_cand", "")):
            seniority_scores.append(self._seniority_score(jr, cr))
        seniority_scores = np.asarray(seniority_scores, dtype=float)

        return np.vstack([text_similarity, overlaps, seniority_scores]).T


def build_feature_transformer(max_features: int = 5000) -> ColumnTransformer:
    """Retorna transformador com TF-IDF (vaga/candidato separados) + one-hot + numéricas."""
    job_text_vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words="portuguese",
    )
    cand_text_vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words="portuguese",
    )
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_engineered = PairwiseTfidfSimilarityAndSkills(max_features=8000)
    # Evita colunas duplicadas entre transformadores
    numeric_cols = [
        col for col in NUMERIC_SOURCE_COLUMNS if col not in CATEGORICAL_COLUMNS
    ]

    transformer = ColumnTransformer(
        transformers=[
            ("job_tfidf", job_text_vec, "job_text"),
            ("cand_tfidf", cand_text_vec, "candidate_text"),
            ("categorical", categorical_transformer, CATEGORICAL_COLUMNS),
            ("numeric_engineered", numeric_engineered, numeric_cols),
        ],
        remainder="drop",
    )
    return transformer


def build_training_pipeline(max_features: int = 5000, **model_kwargs) -> Pipeline:
    """Cria pipeline com transformações e modelo de classificação."""
    transformer = build_feature_transformer(max_features=max_features)
    classifier = LogisticRegression(
        max_iter=model_kwargs.pop("max_iter", 1000),
        class_weight=model_kwargs.pop("class_weight", "balanced"),
        C=model_kwargs.pop("C", 1.0),
        **model_kwargs,
    )
    pipeline = Pipeline(
        steps=[
            ("features", transformer),
            ("classifier", classifier),
        ]
    )
    return pipeline


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa variáveis independentes e rótulo."""
    # Inclui colunas necessárias para features numéricas adicionais
    needed = TEXT_COLUMNS + CATEGORICAL_COLUMNS + [
        col for col in NUMERIC_SOURCE_COLUMNS if col in df.columns
    ]
    # Remove duplicadas preservando ordem
    needed = list(dict.fromkeys(needed))
    X = df[needed].copy()
    y = df["label"].astype(int)
    return X, y
