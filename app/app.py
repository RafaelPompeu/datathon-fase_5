from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_utils import load_pipeline, rank_candidates_for_job
from src.preprocessing import preprocess_dataset
from src.utils import DATA_DIR, MODELS_DIR, POSITIVE_STATUSES

MODEL_PATH = MODELS_DIR / "modelo_decision.pkl"


@st.cache_resource
def _load_model(path: Path):
    try:
        return load_pipeline(path)
    except FileNotFoundError:
        return None


@st.cache_data
def _load_dataset(data_dir: Path) -> pd.DataFrame:
    return preprocess_dataset(data_dir=data_dir, positive_statuses=POSITIVE_STATUSES)


def main() -> None:
    st.set_page_config(page_title="Decision Match AI", layout="wide")
    st.title("Decision – MVP de Matching com IA")

    dataset = _load_dataset(DATA_DIR)
    model = _load_model(MODEL_PATH)

    if model is None:
        st.warning(
            "Modelo não encontrado. Execute `python -m src.train` para treinar "
            "e gerar o arquivo `models/modelo_decision.pkl`."
        )
        st.stop()

    job_options = (
        dataset[["job_id", "titulo_vaga", "cliente"]]
        .drop_duplicates()
        .sort_values("job_id")
        .assign(label=lambda df: df.apply(lambda row: f"{row.job_id} – {row.titulo_vaga} ({row.cliente})", axis=1))
    )

    job_choice = st.selectbox(
        "Selecione uma vaga",
        options=job_options.to_dict("records"),
        format_func=lambda item: item["label"],
    )

    job_id = job_choice["job_id"]
    ranking = rank_candidates_for_job(model, dataset, job_id=job_id, top_k=5)

    cols_to_show = [
        "codigo_profissional",
        "nome_candidato",
        "situacao_candidado",
        "match_proba",
        "nivel_profissional_cand",
        "nivel_ingles_cand",
        "nivel_academico_cand",
        "comentario",
    ]
    tabela = ranking[cols_to_show].rename(
        columns={
            "codigo_profissional": "Código",
            "nome_candidato": "Candidato",
            "situacao_candidado": "Status Original",
            "match_proba": "Score IA",
            "nivel_profissional_cand": "Nível Profissional",
            "nivel_ingles_cand": "Inglês",
            "nivel_academico_cand": "Formação",
            "comentario": "Observações",
        }
    )

    st.subheader("Ranking de candidatos")
    st.dataframe(tabela.style.format({"Score IA": "{:.2%}"}))

    st.markdown("---")
    st.caption(
        "O score reflete a probabilidade estimada pelo modelo de o candidato estar alinhado "
        "à vaga com base no histórico rotulado."
    )


if __name__ == "__main__":
    main()
