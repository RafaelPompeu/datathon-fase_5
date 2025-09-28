from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_utils import (
    load_pipeline,
    rank_candidates_for_job,
    rank_jobs_for_candidate,
)
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


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          :root {
            --primary: #2b6cb0; /* azul mais sóbrio */
            --accent: #38a169;  /* verde para destaques */
            --bg-soft: #f7fafc; /* cinza muito claro */
            --text-muted: #4a5568;
          }
          .hero {
            background: linear-gradient(90deg, rgba(43,108,176,0.10) 0%, rgba(56,161,105,0.10) 100%);
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 12px;
          }
          .hero h1 { margin: 0 0 6px 0; font-size: 1.6rem; color: var(--primary); }
          .hero p { margin: 0; color: var(--text-muted); }
          .card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 14px 16px;
          }
          .pill {
            display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px;
            background: #ebf8ff; color: #2b6cb0; border:1px solid #bee3f8; margin-right: 6px;
          }
          .pill.green { background:#f0fff4; color:#276749; border-color:#c6f6d5; }
          .pill.orange { background:#fffaf0; color:#975a16; border-color:#feebc8; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _styled_table(
    df: pd.DataFrame,
    score_cols: list[str],
    status_col: str | None = None,
    approved_values: tuple[str, ...] = ("Aprovado",),
) -> "pd.io.formats.style.Styler":
    fmt = {col: "{:.2%}" for col in score_cols if col in df.columns}
    style = df.style.format(fmt)
    try:
        style = style.background_gradient(axis=0, subset=[c for c in score_cols if c in df.columns], cmap="Greens")
    except Exception:
        pass

    if status_col and status_col in df.columns:
        def _hl(row):
            if row.get(status_col) in approved_values:
                return ["background-color: #f0fff4; color:#22543d; font-weight:600;"] * len(row)
            return [""] * len(row)
        style = style.apply(_hl, axis=1)
    return style


def main() -> None:
    st.set_page_config(page_title="Decision Match RH", page_icon="🧑‍💼", layout="wide")
    _inject_css()

    st.markdown(
        """
        <div class="hero">
          <h1>🧑‍💼 Decision – Matching de Vagas para RH</h1>
          <p>Compare o resumo do currículo com as descrições das vagas e encontre o melhor encaixe.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    # Formulário simplificado: recomendação apenas pelo resumo do currículo

    tab_jobs, tab_candidate = st.tabs([
        "Ranking por vaga",
        "Recomendação para candidato",
    ])

    with tab_jobs:
        job_choice = st.selectbox(
            "Selecione uma vaga",
            options=job_options.to_dict("records"),
            format_func=lambda item: item["label"],
        )

        job_id = job_choice["job_id"]
        ranking = rank_candidates_for_job(model, dataset, job_id=job_id, top_k=5)

        # Cartão com resumo da vaga
        job_row = dataset[dataset["job_id"] == job_id].iloc[0]
        col_a, col_b, col_c = st.columns([2, 1.2, 1.2])
        with col_a:
            st.markdown(
                f"""
                <div class="card">
                  <strong style="color:#2b6cb0">{job_row['titulo_vaga']}</strong><br/>
                  <span class="pill">Cliente: {job_row.get('cliente','')}</span>
                  <span class="pill green">{job_row.get('tipo_contratacao','')}</span>
                  <span class="pill orange">Prioridade: {job_row.get('prioridade_vaga','')}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_b:
            st.metric("Candidatos avaliados", f"{len(ranking):,}".replace(",", "."))
        with col_c:
            st.metric("Job ID", str(job_id))

        cols_to_show = [
            "codigo_profissional",
            "nome_candidato",
            "situacao_candidato",
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
                "situacao_candidato": "Status Original",
                "match_proba": "Score IA",
                "nivel_profissional_cand": "Nível Profissional",
                "nivel_ingles_cand": "Inglês",
                "nivel_academico_cand": "Formação",
                "comentario": "Observações",
            }
        )

        st.subheader("Ranking de candidatos")
        st.dataframe(_styled_table(tabela, ["Score IA"], status_col="Status Original"))

        st.markdown("---")
        st.caption(
            "O score reflete a probabilidade estimada pelo modelo de o candidato estar alinhado "
            "à vaga com base no histórico rotulado."
        )

    with tab_candidate:
        st.subheader("Cole seu currículo e encontre a melhor vaga")
        with st.form("candidate_form"):
            cv_text = st.text_area(
                "Resumo do currículo (cole aqui seu CV ou resumo)",
                placeholder="Cole aqui seu resumo, experiências e palavras‑chave do seu currículo",
                height=240,
            )
            submitted = st.form_submit_button("🔎 Buscar vaga mais aderente")

        if submitted:
            if not cv_text.strip():
                st.warning("Cole o resumo do currículo para prosseguir.")
                st.stop()
            profile = {
                "cv_pt": cv_text.strip(),
            }

            try:
                recommendations = rank_jobs_for_candidate(
                    model,
                    dataset,
                    candidate_profile=profile,
                    top_k=1,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                display_cols = [
                    "job_id",
                    "titulo_vaga",
                    "cliente",
                    "tipo_contratacao",
                    "prioridade_vaga",
                    "nivel_profissional_req",
                    "match_score",
                    "match_proba",
                    "text_similarity",
                ]
                available_cols = [col for col in display_cols if col in recommendations.columns]
                info = recommendations.loc[:, available_cols].rename(
                    columns={
                        "job_id": "Vaga",
                        "titulo_vaga": "Título",
                        "cliente": "Cliente",
                        "tipo_contratacao": "Contratação",
                        "prioridade_vaga": "Prioridade",
                        "nivel_profissional_req": "Senioridade desejada",
                        "match_score": "Score combinado",
                        "match_proba": "Score histórico",
                        "text_similarity": "Similaridade texto",
                    }
                )
                best = info.iloc[0]
                best_score = float(best.get("Score combinado", best.get("Score histórico", 0)))
                a, b, c = st.columns([1.2, 1, 1])
                a.metric("Score combinado", f"{best_score:.2%}")
                b.metric("Score histórico", f"{float(best['Score histórico']):.2%}")
                c.metric("Similaridade", f"{float(best['Similaridade texto']):.2%}")

                st.markdown(
                    f"""
                    <div class="card" style="margin-top:8px;">
                      <strong style="color:#2b6cb0">{best['Título']}</strong><br/>
                      <span class="pill">Cliente: {best.get('Cliente','')}</span>
                      <span class="pill green">{best.get('Contratação','')}</span>
                      <span class="pill orange">Prioridade: {best.get('Prioridade','')}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.success("Encontramos a vaga mais aderente ao seu perfil!")
                st.dataframe(_styled_table(info, ["Score combinado", "Score histórico", "Similaridade texto"]))


if __name__ == "__main__":
    main()
