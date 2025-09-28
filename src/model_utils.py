"""Funções utilitárias para modelos treinados."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .feature_engineering import split_features_target
from .utils import MIN_JOB_WORDS


DEFAULT_CATEGORY_VALUE = "Não informado"
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
DEFAULT_SENIORITY_RANK = 2


def _stringify_value(value: Any) -> str:
    """Converte diferentes tipos em string legível."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value if item)
    return str(value)


def _normalize_category(value: Any, default: str = DEFAULT_CATEGORY_VALUE) -> str:
    text = _stringify_value(value).strip()
    return text if text else default


def _build_candidate_text(profile: Dict[str, Any]) -> str:
    """Usa apenas o resumo/trecho do currículo informado pelo candidato."""
    text = _stringify_value(profile.get("cv_pt") or profile.get("cv_text"))
    return (text or "").strip().lower()


def _compute_text_similarity(candidate_text: str, job_texts: List[str]) -> pd.Series:
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    texts = [candidate_text] + [text or "" for text in job_texts]
    matrix = vectorizer.fit_transform(texts)
    candidate_vec = matrix[0]
    job_vecs = matrix[1:]
    similarities = cosine_similarity(candidate_vec, job_vecs).flatten()
    return pd.Series(similarities)


def _tokenize_skills(text: Any) -> Set[str]:
    raw = _stringify_value(text).lower()
    if not raw:
        return set()
    tokens = {
        token
        for token in re.split(r"[\s,;\-/\\\n]+", raw)
        if len(token) >= 3 and token.isalpha()
    }
    return tokens


# Vocabulários simples para identificar domínios
TECH_TOKENS = {
    "python", "django", "flask", "fastapi", "api", "apis", "rest",
    "sql", "nosql", "postgres", "mysql", "mongodb", "etl", "elt",
    "spark", "hadoop", "airflow", "kafka", "docker", "kubernetes",
    "ci", "cd", "devops", "git", "aws", "gcp", "azure", "ml",
    "machine", "learning", "dados", "data", "engineer", "backend",
    "frontend", "web", "microservices", "microserviços",
}
STOCK_TOKENS = {
    "estoque", "almoxarifado", "almoxarife", "expedição", "expedicao",
    "logística", "logistica", "separação", "separacao", "conferência",
    "conferencia", "recebimento", "armazém", "armazem", "carga",
    "descarga", "inventário", "inventario",
}


def _contains_any(tokens: Set[str], vocab: Set[str]) -> bool:
    return any(tok in vocab for tok in tokens)


def _domain_factor(candidate_text: str, job_text: str) -> float:
    c_tokens = _tokenize_skills(candidate_text)
    j_tokens = _tokenize_skills(job_text)
    cand_is_tech = _contains_any(c_tokens, TECH_TOKENS)
    job_is_tech = _contains_any(j_tokens, TECH_TOKENS)
    job_is_stock = _contains_any(j_tokens, STOCK_TOKENS)
    # Conflito forte: CV técnico x vaga de estoque
    if cand_is_tech and job_is_stock and not job_is_tech:
        return 0.5  # penaliza 50%
    # Alinhamento técnico: leve reforço
    if cand_is_tech and job_is_tech:
        return 1.15
    return 1.0


def _collect_candidate_skill_text(profile: Dict[str, Any]) -> str:
    """Considera apenas o texto do resumo do currículo."""
    return _stringify_value(profile.get("cv_pt") or profile.get("cv_text") or "").strip()


def _normalize_seniority_value(value: Any) -> str:
    return _stringify_value(value).strip().lower()


def _seniority_rank(value: Any) -> int:
    normalized = _normalize_seniority_value(value)
    if not normalized:
        return -1
    return SENIORITY_RANK.get(normalized, DEFAULT_SENIORITY_RANK)


def _seniority_score(job_value: Any, candidate_rank: int) -> float:
    job_rank = _seniority_rank(job_value)
    if candidate_rank < 0 and job_rank < 0:
        return 0.5
    if candidate_rank < 0:
        return 0.6
    if job_rank < 0:
        return 0.6
    distance = abs(job_rank - candidate_rank)
    return max(0.0, 1.0 - 0.25 * distance)


def _prepare_jobs_for_candidate(data: pd.DataFrame, profile: Dict[str, Any]) -> pd.DataFrame:
    required_columns = [
        "job_id",
        "job_text",
        "titulo_vaga",
        "cliente",
        "tipo_contratacao",
        "prioridade_vaga",
    ]
    missing = set(required_columns).difference(data.columns)
    if missing:
        raise ValueError(
            "Dataset de referência não possui as colunas necessárias: "
            + ", ".join(sorted(missing))
        )

    candidate_text = _build_candidate_text(profile)
    if not candidate_text:
        raise ValueError(
            "Informe ao menos um texto do currículo, conhecimentos ou experiências do candidato."
        )

    optional_columns = [
        "nivel_profissional_req",
        "competencias_tecnicas",
        "principais_atividades",
    ]
    selected_columns = list(required_columns) + [col for col in optional_columns if col in data.columns]
    jobs = data.drop_duplicates(subset=["job_id"]).loc[:, selected_columns].copy()

    jobs["codigo_profissional"] = _stringify_value(profile.get("codigo_profissional")) or "candidato_manual"
    jobs["nome_candidato"] = _stringify_value(profile.get("nome_candidato") or profile.get("nome")) or "Candidato"
    jobs["local_candidato"] = _stringify_value(profile.get("local_candidato"))
    jobs["nivel_academico_cand"] = _stringify_value(profile.get("nivel_academico_cand"))
    jobs["conhecimentos_tecnicos"] = _stringify_value(profile.get("conhecimentos_tecnicos"))
    jobs["experiencias"] = _stringify_value(profile.get("experiencias"))
    jobs["cv_pt"] = _stringify_value(profile.get("cv_pt") or profile.get("cv_text"))
    jobs["nivel_profissional_cand"] = _normalize_category(profile.get("nivel_profissional_cand"))
    jobs["nivel_ingles_cand"] = _normalize_category(profile.get("nivel_ingles_cand"))
    jobs["situacao_candidado"] = "Novo"
    jobs["candidate_text"] = candidate_text

    jobs["job_text"] = jobs["job_text"].fillna("")
    job_word_counts = jobs["job_text"].str.split().str.len()
    valid_mask = job_word_counts >= MIN_JOB_WORDS
    if valid_mask.any():
        jobs = jobs[valid_mask].copy()
    jobs["job_text"] = jobs["job_text"].fillna("")

    jobs["combined_text"] = (jobs["job_text"] + " \n " + jobs["candidate_text"]).str.strip()

    jobs = jobs[jobs["combined_text"].str.len() > 0]
    if jobs.empty:
        raise ValueError("Não há vagas disponíveis para recomendar no momento.")

    for column in ["tipo_contratacao", "prioridade_vaga", "nivel_profissional_cand", "nivel_ingles_cand"]:
        jobs[column] = jobs[column].fillna(DEFAULT_CATEGORY_VALUE)

    return jobs


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


def rank_jobs_for_candidate(
    pipeline,
    data: pd.DataFrame,
    candidate_profile: Dict[str, Any],
    top_k: int = 5,
) -> pd.DataFrame:
    """Gera todas as combinações candidato-vaga e retorna as melhores pontuações."""
    prepared = _prepare_jobs_for_candidate(data, candidate_profile)
    scores = predict_probabilities(pipeline, prepared)
    prepared["match_proba"] = scores

    text_similarity = _compute_text_similarity(
        candidate_text=prepared["candidate_text"].iloc[0],
        job_texts=prepared["job_text"].tolist(),
    )
    prepared["text_similarity"] = text_similarity.values

    candidate_skill_tokens = _tokenize_skills(_collect_candidate_skill_text(candidate_profile))
    if "competencias_tecnicas" in prepared.columns:
        job_skill_text = prepared["competencias_tecnicas"].fillna("")
    else:
        job_skill_text = pd.Series(["" for _ in range(len(prepared))], index=prepared.index)
    if "principais_atividades" in prepared.columns:
        job_skill_text = job_skill_text.str.cat(prepared["principais_atividades"].fillna(""), sep=" ")
    job_skill_text = job_skill_text.str.cat(prepared["job_text"], sep=" ").str.strip()

    if candidate_skill_tokens:
        prepared["skill_overlap"] = job_skill_text.apply(
            lambda text: (
                len(candidate_skill_tokens & _tokenize_skills(text))
                / max(len(candidate_skill_tokens), 1)
            )
        )
    else:
        prepared["skill_overlap"] = 0.0

    # Score combinado (foco no resumo + penalização de domínio)
    base_score = 0.45 * prepared["text_similarity"] + 0.55 * prepared["match_proba"]
    domain = prepared.apply(lambda r: _domain_factor(r.get("candidate_text", ""), r.get("job_text", "")), axis=1)
    prepared["match_score"] = base_score * domain

    return prepared.sort_values("match_score", ascending=False).head(top_k)
