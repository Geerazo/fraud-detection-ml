Fraud Detection in Financial Transactions

![CI](https://github.com/Geerazo/fraud-detection-ml/actions/workflows/ci.yml/badge.svg)

Pipeline reproducible para detección de fraude (Kaggle Worldline). Incluye preprocesamiento, balanceo SMOTE+under, XGBoost y evaluación, más API de inferencia (FastAPI) y ejemplo de consumo desde Power BI.

📦 Requisitos

Python 3.12

Windows probado (PowerShell). Funciona en Linux/macOS con pequeños cambios de activación del venv.

🗂️ Estructura (resumen)
fraud-detection-ml/
 ├─ data/
 │   ├─ raw/creditcard.csv
 │   └─ processed/{train.csv, test.csv, train_bal.csv}
 ├─ models/{model_xgb.json, feature_order.json, scaler.pkl?}
 ├─ reports/{metrics.json, figures/}
 └─ src/{data, models, serving}

⬇️ Datos

Descarga desde Kaggle (Worldline) y deja el CSV en data/raw/creditcard.csv.

⚙️ Pasos rápidos
# 0) Activar venv e instalar
pip install -r requirements.txt
python -m pytest

# 1) Preprocesar (requiere data/raw/creditcard.csv)
python -m src.data.preprocess --input data/raw/creditcard.csv --outdir data/processed --test-size 0.25

# 2) Balancear (sólo TRAIN)
python -m src.data.balance --train data/processed/train.csv --out data/processed/train_bal.csv --smote 0.2 --under 0.5

# 3) Entrenar (guarda modelo y feature_order.json)
python -m src.models.train --train data/processed/train_bal.csv --model-dir models

# 4) Evaluar
python -m src.models.evaluate --test data/processed/test.csv --model models/model_xgb.json --scaler models/scaler.pkl --out reports


Nota: evaluate crea reports/metrics.json. Si quieres curvas, asegúrate de que exista reports/figures/.

📈 Resultados (reales)

ROC-AUC: 0.9829

PR-AUC: 0.8289

Umbral operativo sugerido: 0.25

Recall (fraude): 0.8699

Precision (fraude): 0.2069

F1: 0.3344

Matriz de confusión (test): TN=70669, FP=410, FN=16, TP=107

Comentario: Con umbral 0.25 se captura ~87% del fraude con FPR ≈ 0.58%. Recomendado para colas de revisión priorizadas y ajuste fino por costos.

🧠 Model Card (resumen)

Objetivo: priorizar transacciones para revisión antifraude.

Datos: Kaggle Worldline (Europa, 2013); variables PCA V1–V28 + Time/Amount.

Métricas: ROC-AUC 0.983, PR-AUC 0.829, Recall@0.25 0.87, FPR ~0.58%.

Riesgos: dataset shift; PCA dificulta interpretación; falsos positivos.

Controles: monitoreo de drift, re-entrenamiento periódico, human-in-the-loop, derecho a apelación.

Limitaciones: set público; no refleja patrones actuales de tu organización.

🔄 Reproducibilidad & CI

Python: 3.12

Crear entorno: python -m venv .venv && .\.venv\Scripts\Activate.ps1

Instalar: pip install -r requirements.txt

Tests: pytest -q

Linter: flake8 src

CI (sugerido): GitHub Actions (lint + tests en cada push/PR)

🧩 Troubleshooting

422 (Validation Error): faltan columnas o Time/Amount negativos (datos escalados). Usa RAW de data/raw/*.csv.

500: revisa detail y logs de Uvicorn (orden de columnas o tipos).

Power BI no conecta: usar http://localhost:8000, borrar Data Source Settings, Anonymous + Public, ignorar Privacy Levels, revisar firewall.

🤝 Contribuir

Issues y PRs bienvenidos: nuevos modelos (LightGBM, CatBoost), explicabilidad (SHAP), time-aware split, coste económico/umbral, despliegues (Docker/Render/Railway).