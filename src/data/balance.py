import argparse
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from pathlib import Path

def main(args):
    train = pd.read_csv(args.train)
    assert "Class" in train.columns, "Se requiere columna 'Class'"
    X = train.drop(columns=["Class"])
    y = train["Class"]

    pipeline = Pipeline(steps=[
        ("smote", SMOTE(sampling_strategy=args.smote, random_state=42)),
        ("under", RandomUnderSampler(sampling_strategy=args.under, random_state=42))
    ])
    Xb, yb = pipeline.fit_resample(X, y)
    out = Xb.copy()
    out["Class"] = yb
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[OK] Train balanceado guardado en {args.out} (shape={out.shape})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Ruta a train.csv preprocesado")
    ap.add_argument("--out", required=True, help="Ruta de salida para train_bal.csv")
    ap.add_argument("--smote", type=float, default=0.2, help="Ratio minoría tras SMOTE")
    ap.add_argument("--under", type=float, default=0.5, help="Ratio mayoría tras under-sampling")
    main(ap.parse_args())
