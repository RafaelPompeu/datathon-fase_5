"""Carregamento, extração e pré-processamento dos dados Decision."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from .utils import DATA_DIR, POSITIVE_STATUSES, ensure_directories, setup_logging

try:
    from google.cloud import bigquery
except ImportError:  # pragma: no cover - dependência opcional
    bigquery = None


DEFAULT_QUERY = """
WITH pros_last AS (
  SELECT
    job_id,
    codigo_profissional,
    ANY_VALUE(nome_prospect) AS nome_prospect,
    ANY_VALUE(recrutador)    AS recrutador,
    ANY_VALUE(comentario)    AS comentario,
    MAX(ultima_atualizacao)  AS dt_last_update,
    ARRAY_AGG(situacao_candidado IGNORE NULLS ORDER BY ultima_atualizacao DESC LIMIT 1)[OFFSET(0)] AS situacao_candidado_last
  FROM datathon-470123.silver.vw_prospects_flat 
  GROUP BY 1,2
)
SELECT
  v.job_id, v.titulo_vaga, v.cliente, v.tipo_contratacao, v.analista_responsavel, v.prioridade_vaga,
  v.data_inicial, v.data_final,
  v.competencias_tecnicas, v.principais_atividades, a.cv_pt, a.cv_en,
  v.nivel_ingles_req, v.nivel_espanhol_req, v.nivel_academico_req, v.nivel_profissional_req,
  a.codigo_profissional, a.nome_candidato, a.email_candidato, a.local_candidato,
  a.nivel_ingles_cand, a.nivel_espanhol_cand, a.nivel_academico_cand, a.nivel_profissional_cand,
  a.conhecimentos_tecnicos, a.experiencias, a.cargo_atual_cand,
  p.situacao_candidado_last,
  p.recrutador, p.comentario, p.dt_last_update
FROM datathon-470123.silver.vw_vagas_flat v
LEFT JOIN pros_last p
  ON p.job_id = v.job_id
LEFT JOIN datathon-470123.silver.vw_applicant_flat a
  ON a.codigo_profissional = p.codigo_profissional
WHERE p.situacao_candidado_last IN (
    'Aprovado',
    'Não Aprovado pelo Requisitante',
    'Proposta Aceita',
    'Recusado',
    'Documentação PJ',
    'Documentação CLT'
  )
  AND a.cv_pt IS NOT NULL
"""


def _read_json(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_json(path)


def load_raw_data(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega arquivos JSON (vagas, prospects, applicants)."""
    vacancies = _read_json(data_dir / "vagas.json")
    prospects = _read_json(data_dir / "prospects.json")
    applicants = _read_json(data_dir / "applicants.json")
    return vacancies, prospects, applicants


def _write_json(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(records, fp, ensure_ascii=False, indent=2)


def fetch_bigquery_dataframe(
    project_id: str,
    query: str,
    credentials_path: str | None = None,
) -> pd.DataFrame:
    """Executa query no BigQuery e retorna DataFrame."""
    if bigquery is None:
        raise ImportError(
            "google-cloud-bigquery não está instalado. Use `pip install google-cloud-bigquery`."
        )

    if credentials_path:
        client = bigquery.Client.from_service_account_json(credentials_path, project=project_id)
    else:
        client = bigquery.Client(project=project_id)

    logging.info("Executando query no BigQuery...")
    job = client.query(query)
    df = job.result().to_dataframe()
    logging.info("Obtidas %d linhas do BigQuery", len(df))
    return df


VACANCY_COLUMNS = [
    "job_id",
    "titulo_vaga",
    "cliente",
    "tipo_contratacao",
    "analista_responsavel",
    "prioridade_vaga",
    "data_inicial",
    "data_final",
    "competencias_tecnicas",
    "principais_atividades",
    "nivel_ingles_req",
    "nivel_espanhol_req",
    "nivel_academico_req",
    "nivel_profissional_req",
]

APPLICANT_COLUMNS = [
    "codigo_profissional",
    "nome_candidato",
    "email_candidato",
    "local_candidato",
    "nivel_ingles_cand",
    "nivel_espanhol_cand",
    "nivel_academico_cand",
    "nivel_profissional_cand",
    "conhecimentos_tecnicos",
    "experiencias",
    "cargo_atual_cand",
    "cv_pt",
    "cv_en",
]

PROSPECT_COLUMNS = [
    "job_id",
    "codigo_profissional",
    "situacao_candidado_last",
    "recrutador",
    "comentario",
    "dt_last_update",
]

DATE_COLUMNS = ["data_inicial", "data_final", "dt_last_update"]


def dataframe_to_decision_json(df: pd.DataFrame, output_dir: Path = DATA_DIR) -> None:
    """Converte DataFrame da query (join) em três arquivos JSON."""
    output_dir = Path(output_dir)
    ensure_directories([output_dir])

    df = df.copy()
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")

    df = df.where(pd.notna(df), None)

    vacancies = (
        df[VACANCY_COLUMNS]
        .drop_duplicates(subset=["job_id"])
        .sort_values("job_id")
        .to_dict(orient="records")
    )
    applicants = (
        df[APPLICANT_COLUMNS]
        .drop_duplicates(subset=["codigo_profissional"])
        .sort_values("codigo_profissional")
        .to_dict(orient="records")
    )
    prospects = (
        df[PROSPECT_COLUMNS]
        .drop_duplicates(subset=["job_id", "codigo_profissional"], keep="last")
        .sort_values(["job_id", "codigo_profissional"])
        .to_dict(orient="records")
    )

    _write_json(vacancies, output_dir / "vagas.json")
    _write_json(applicants, output_dir / "applicants.json")
    _write_json(prospects, output_dir / "prospects.json")
    logging.info(
        "JSONs gerados em %s (vagas=%d, applicants=%d, prospects=%d)",
        output_dir,
        len(vacancies),
        len(applicants),
        len(prospects),
    )


def _stringify_list(column: pd.Series) -> pd.Series:
    return column.apply(lambda value: ", ".join(value) if isinstance(value, list) else value)


def preprocess_dataset(
    data_dir: Path = DATA_DIR,
    positive_statuses: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Une e higieniza datasets criando coluna de rótulo binário."""
    positive = set(positive_statuses or POSITIVE_STATUSES)
    vacancies, prospects, applicants = load_raw_data(data_dir)

    # Normaliza listas textuais
    vacancies["competencias_tecnicas"] = _stringify_list(vacancies.get("competencias_tecnicas"))
    applicants["conhecimentos_tecnicos"] = _stringify_list(applicants.get("conhecimentos_tecnicos"))

    merged = (
        prospects.merge(vacancies, on="job_id", how="left", suffixes=("_prospect", "_vaga"))
        .merge(applicants, on="codigo_profissional", how="left", suffixes=("", "_app"))
    )

    merged.dropna(subset=["situacao_candidado_last", "cv_pt"], inplace=True)
    merged["situacao_candidado_last"] = merged["situacao_candidado_last"].str.strip()
    merged["situacao_candidado"] = merged["situacao_candidado_last"].fillna("")
    merged["label"] = merged["situacao_candidado"].isin(positive).astype(int)

    merged["competencias_tecnicas"] = merged["competencias_tecnicas"].fillna("")
    merged["principais_atividades"] = merged["principais_atividades"].fillna("")

    merged["conhecimentos_tecnicos"] = merged["conhecimentos_tecnicos"].fillna("")
    merged["experiencias"] = merged["experiencias"].fillna("")
    merged["cv_pt"] = merged["cv_pt"].fillna("")

    merged["job_text"] = (
        merged["titulo_vaga"].fillna("")
        + " "
        + merged["cliente"].fillna("")
        + " "
        + merged["competencias_tecnicas"]
        + " "
        + merged["principais_atividades"]
        + " "
        + merged["nivel_profissional_req"].fillna("")
    ).str.lower()

    merged["candidate_text"] = (
        merged["cv_pt"]
        + " "
        + merged["conhecimentos_tecnicos"]
        + " "
        + merged["experiencias"]
        + " "
        + merged["nivel_profissional_cand"].fillna("")
    ).str.lower()

    merged["combined_text"] = (merged["job_text"] + " \n " + merged["candidate_text"]).str.strip()

    categorical_cols = [
        "tipo_contratacao",
        "prioridade_vaga",
        "nivel_profissional_cand",
        "nivel_ingles_cand",
    ]
    for col in categorical_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna("Não informado")

    merged = merged[merged["combined_text"].str.len() > 0]

    merged = merged.drop_duplicates(subset=["job_id", "codigo_profissional"], keep="last")
    merged.reset_index(drop=True, inplace=True)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta dados do BigQuery para JSON e/ou preprocessa dataset")
    parser.add_argument("--project-id", help="ID do projeto GCP com os datasets Decision")
    parser.add_argument(
        "--credentials-path",
        help="Caminho para o JSON da service account (opcional se já tiver GOOGLE_APPLICATION_CREDENTIALS)",
    )
    parser.add_argument(
        "--query-path",
        type=Path,
        help="Arquivo com a query personalizada (default: query oficial do Datathon)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Diretório onde os JSON serão salvos",
    )
    parser.add_argument(
        "--only-preprocess",
        action="store_true",
        help="Apenas realiza o pré-processamento dos JSON existentes e imprime estatísticas",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    if not args.only_preprocess:
        if not args.project_id:
            raise SystemExit("--project-id é obrigatório para extrair dados do BigQuery")
        query = DEFAULT_QUERY
        if args.query_path:
            query = args.query_path.read_text(encoding="utf-8")

        df = fetch_bigquery_dataframe(
            project_id=args.project_id,
            query=query,
            credentials_path=args.credentials_path,
        )
        dataframe_to_decision_json(df, output_dir=args.output_dir)

    dataset = preprocess_dataset(data_dir=args.output_dir)
    logging.info("Dataset preprocessado: %d linhas", len(dataset))
    print(dataset.head().to_string())


if __name__ == "__main__":
    main()
