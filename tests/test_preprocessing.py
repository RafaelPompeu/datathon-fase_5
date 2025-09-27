from src.preprocessing import preprocess_dataset


def test_preprocess_dataset_columns():
    df = preprocess_dataset()
    expected_columns = {
        "job_id",
        "codigo_profissional",
        "situacao_candidato",
        "job_text",
        "candidate_text",
        "combined_text",
        "label",
    }
    assert expected_columns.issubset(df.columns)
    # garante que temos tanto rótulos positivos quanto negativos na amostra sintética
    assert set(df["label"]) == {0, 1}
    # não deve haver linhas duplicadas por job/candidato
    duplicates = df.duplicated(subset=["job_id", "codigo_profissional"])
    assert duplicates.sum() == 0
