import argparse
from pathlib import Path
import json
import pandas as pd
from xgboost import XGBClassifier


def main(args):
    train_path = Path(args.train)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_path)
    assert "Class" in df.columns, "El CSV de entrenamiento debe tener columna 'Class'."

    X = df.drop(columns=["Class"])
    y = df["Class"]

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=1.0,  # ya está balanceado (SMOTE+under)
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss",
    )

    xgb.fit(X, y)

    # --- Guardar modelo ---
    model_path = model_dir / "model_xgb.json"
    xgb.save_model(str(model_path))

    # --- Guardar orden de features (blindaje para la API) ---
    feature_order = X.columns.tolist()
    with open(model_dir / "feature_order.json", "w", encoding="utf-8") as f:
        json.dump(feature_order, f, ensure_ascii=False, indent=2)

    print(f"[OK] Modelo guardado en {model_path}")
    print(f"[OK] feature_order guardado en {model_dir / 'feature_order.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Ruta a data/processed/train_bal.csv")
    ap.add_argument("--model_dir", default="models", help="Carpeta donde guardar el modelo")
    main(ap.parse_args())
