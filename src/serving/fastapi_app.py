from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List
import pandas as pd
import joblib
from xgboost import XGBClassifier
import json
import io

app = FastAPI(title="Fraud Scoring API", version="0.1.0")

# --- Rutas de artefactos ---
MODEL_PATH = Path("models/model_xgb.json")
SCALER_PATH = Path("models/scaler.pkl")              # opcional, no usado por XGB en este proyecto
FEATURE_ORDER_PATH = Path("models/feature_order.json")

# --- Umbral operativo de decisión ---
THRESHOLD = 0.25

# --- CORS (útil para Power BI / frontends) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # en producción, limita dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Esquemas de entrada/salida (validación amigable) ---
class TransactionInput(BaseModel):
    # Validaciones mínimas: no negativos en tiempo y monto
    Time: float = Field(ge=0)
    Amount: float = Field(ge=0)
    # Componentes PCA
    V1: float;  V2: float;  V3: float;  V4: float;  V5: float;  V6: float;  V7: float
    V8: float;  V9: float;  V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time": 10000, "Amount": 149.5,
                "V1": 0.1, "V2": -1.2, "V3": 0.05, "V4": 0.3, "V5": -0.7, "V6": 0.2, "V7": 0.1,
                "V8": 0.0, "V9": 0.2, "V10": -0.1, "V11": 0.6, "V12": 0.1, "V13": 0.2, "V14": -0.05,
                "V15": 0.1, "V16": 0.2, "V17": -0.3, "V18": 0.4, "V19": 0.0, "V20": -0.2, "V21": 0.1,
                "V22": 0.0, "V23": -0.1, "V24": 0.2, "V25": 0.1, "V26": 0.0, "V27": -0.05, "V28": 0.3
            }
        }
    }

class ScoreResponse(BaseModel):
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    label: int = Field(..., description="1=fraude, 0=no fraude (según threshold)")
    threshold: float = THRESHOLD

# --- Carga de artefactos al iniciar ---
model = XGBClassifier()
if not MODEL_PATH.exists():
    raise RuntimeError(f"No se encontró el modelo en {MODEL_PATH}. Entrena antes de servir.")
model.load_model(str(MODEL_PATH))

# (no usamos scaler con XGB, se deja por si migras)
scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None

# Orden de columnas: preferir el guardado por train.py; si no existe, usar fallback seguro
if FEATURE_ORDER_PATH.exists():
    FEATURE_ORDER: List[str] = json.loads(FEATURE_ORDER_PATH.read_text(encoding="utf-8"))
else:
    FEATURE_ORDER = (
        ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "hour_of_day", "day_of_week", "is_weekend"]
    )

# --- Helpers ---
def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Time" in df.columns:
        df["hour_of_day"] = ((df["Time"] // 3600) % 24).astype(int)
        df["day_of_week"] = ((df["Time"] // (3600 * 24)) % 7).astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df

def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica FE y devuelve X alineado y tipeado como en entrenamiento."""
    df = engineer_time_features(df)
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")
    X = df[FEATURE_ORDER].astype("float32")
    return X

# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score", response_model=ScoreResponse)
def score(payload: TransactionInput):
    try:
        X = prepare_X(pd.DataFrame([payload.model_dump()]))
        proba = float(model.predict_proba(X)[:, 1][0])
        label = int(proba >= THRESHOLD)
        return {"fraud_probability": proba, "label": label, "threshold": THRESHOLD}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {type(e).__name__}: {e}")

@app.post("/score-batch")
def score_batch(file: UploadFile = File(...)):
    """
    Recibe un CSV (multipart/form-data) con columnas: Time, Amount, V1..V28.
    Devuelve registros con fraud_probability y label.
    """
    try:
        content = file.file.read()
        df = pd.read_csv(io.BytesIO(content))
        X = prepare_X(df)
        proba = model.predict_proba(X)[:, 1]
        label = (proba >= THRESHOLD).astype(int)
        out = df.copy()
        out["fraud_probability"] = proba
        out["label"] = label
        return out.to_dict(orient="records")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch scoring error: {type(e).__name__}: {e}")

@app.post("/score-batch-json", response_model=List[ScoreResponse])
def score_batch_json(payloads: List[TransactionInput]):
    """
    Recibe una lista JSON de transacciones (mismas claves que TransactionInput).
    Devuelve lista de ScoreResponse.
    """
    try:
        df = pd.DataFrame([p.model_dump() for p in payloads])
        X = prepare_X(df)
        proba = model.predict_proba(X)[:, 1]
        label = (proba >= THRESHOLD).astype(int)
        return [
            ScoreResponse(fraud_probability=float(p), label=int(l), threshold=THRESHOLD)
            for p, l in zip(proba, label)
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch JSON scoring error: {type(e).__name__}: {e}")

# --- Arranque directo en Windows (opcional) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.fastapi_app:app", host="127.0.0.1", port=8000, reload=True, log_level="debug")
