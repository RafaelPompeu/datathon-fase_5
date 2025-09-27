# Decision Match AI – Datathon Fase 5

MVP completo para apoiar o time da Decision no processo de recrutamento. O projeto reúne um pipeline de Machine Learning (com todas as etapas solicitadas), um modelo serializado e uma aplicação Streamlit que ranqueia candidatos para cada vaga com base no histórico rotulado.

## Estrutura do repositório

```
├── app/                   # Interface Streamlit
│   └── app.py
├── data/decision/         # Dados JSON (vagas, prospects, applicants) inspirados na query oficial do BQ
├── models/                # Modelos e métricas serializadas
├── notebooks/             # Experimentos e EDA (opcional)
├── reports/               # Relatórios gerados pelo treino (predições, dataset preparado)
├── src/
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── model_utils.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── utils.py
│   └── __init__.py
├── tests/                 # Testes unitários (pytest)
└── requirements.txt
```

## Stack utilizada
- Python 3.12
- Pandas, NumPy
- Scikit-learn (TF-IDF + Logistic Regression)
- Streamlit
- Joblib
- Pytest

## Como preparar o ambiente
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Pipeline de Machine Learning
1. **Pré-processamento** (`src/preprocessing.py`): leitura dos JSON (vagas, prospects, applicants), merge das tabelas, limpeza de textos e criação do rótulo binário com base nos status.
2. **Engenharia de features** (`src/feature_engineering.py`): TF-IDF na concatenação vaga + candidato, one-hot para atributos categóricos (tipo de contratação, prioridade, senioridade, nível de inglês).
3. **Treinamento e validação** (`src/train.py`): separação treino/validação estratificada, ajuste do modelo (Logistic Regression), cálculo de métricas (accuracy, precision, recall, F1, ROC-AUC) e serialização do pipeline.
4. **Avaliação** (`src/evaluate.py`): relatório de classificação e matriz de confusão sobre o dataset completo.

Artefatos gravados automaticamente:
- `models/modelo_decision.pkl` – pipeline (transformações + modelo) serializado.
- `models/metrics.json` – métricas da validação hold-out.
- `reports/validation_predictions.csv` – probabilidades previstas na validação.
- `reports/dataset_preparado.csv` – dataset unificado após pré-processamento.

## Como treinar novamente
```bash
python3 -m src.train \
  --data-dir data/decision \
  --model-path models/modelo_decision.pkl \
  --metrics-path models/metrics.json \
  --test-size 0.3 \
  --random-state 42
```
(Os parâmetros são opcionais; os valores acima são os padrões do script.)

## Como gerar os JSON a partir do BigQuery
```bash
python3 -m src.preprocessing \
  --project-id datathon-470123 \
  --credentials-path /caminho/para/service-account.json \
  --query-path minha_query.sql \
  --output-dir data/decision
```
- Se `--query-path` for omitido, a query padrão do Datathon (presente no código) será utilizada.
- Utilize `--only-preprocess` para apenas transformar os JSON já salvos em dataset pré-processado (sem acessar o BQ).
- É necessário instalar `google-cloud-bigquery` (incluso em `requirements.txt`).

## Como executar a avaliação completa
```bash
python3 -m src.evaluate \
  --model-path models/modelo_decision.pkl \
  --data-dir data/decision \
  --output reports/evaluation_report.json
```

## Como rodar a aplicação Streamlit
```bash
streamlit run app/app.py
```
A aplicação lê o modelo serializado, permite escolher uma vaga e exibe o ranking dos candidatos com o score previsto.

## Como rodar os testes
```bash
python3 -m pytest
```

## Próximos passos sugeridos
1. Incorporar análises de fit cultural e engajamento com dados adicionais (ex.: respostas de entrevistas).
2. Experimentar modelos baseados em embeddings pré-treinados (Sentence Transformers) para comparação com o baseline TF-IDF.
3. Ampliar o conjunto de features numéricas (tempo de experiência, compatibilidade de senioridade) e criar explicações para o ranking (SHAP/LIME).
4. Publicar a aplicação no Streamlit Cloud ou infraestrutura interna para feedback de recrutadores.
