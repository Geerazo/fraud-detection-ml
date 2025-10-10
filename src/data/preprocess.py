import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Time" in df.columns:
        df["hour_of_day"] = ((df["Time"] // 3600) % 24).astype(int)
        df["day_of_week"] = ((df["Time"] // (3600*24)) % 7).astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df

def winsorize_amount(s: pd.Series, lower_q=0.01, upper_q=0.99):
    lo, hi = s.quantile(lower_q), s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)

def main(args):
    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    # Basic sanity
    assert "Class" in df.columns, "El dataset debe contener la columna 'Class'."
    # Winsorize Amount si existe
    if "Amount" in df.columns:
        df["Amount"] = winsorize_amount(df["Amount"])

    # Engineer time features
    df = engineer_time_features(df)

    # Split
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, stratify=df["Class"], random_state=42
    )

    # Scale numeric columns (except target)
    target = "Class"
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])
    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    # Save outputs
    train_out = outdir / "train.csv"
    test_out = outdir / "test.csv"
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    joblib.dump(scaler, model_dir / "scaler.pkl")
    print(f"[OK] Guardados: {train_out}, {test_out} y scaler.pkl en {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Ruta a creditcard.csv")
    parser.add_argument("--outdir", default="data/processed", help="Carpeta salida")
    parser.add_argument("--model_dir", default="models", help="Dónde guardar scaler.pkl")
    parser.add_argument("--test_size", type=float, default=0.25)
    args = parser.parse_args()
    main(args)
