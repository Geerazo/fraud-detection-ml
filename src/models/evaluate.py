import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier
from src.utils.metrics import dump_metrics

def load_xgb(path: str) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(path)
    return model

def plot_roc(y_true, y_score, out_path):
    # ROC via sklearn (only AUC here) — keeping plot simple with default styles
    from sklearn.metrics import RocCurveDisplay
    disp = RocCurveDisplay.from_predictions(y_true, y_score)
    plt.title("ROC Curve")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_pr(y_true, y_score, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve (AUC={pr_auc:.3f})")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return pr_auc

def main(args):
    test = pd.read_csv(args.test)
    assert "Class" in test.columns
    y = test["Class"].values
    X = test.drop(columns=["Class"])

    scaler = None
    if args.scaler and Path(args.scaler).exists():
        scaler = joblib.load(args.scaler)
        # En este punto el test ya está escalado desde preprocess; scaler se usa para servir API

    model = load_xgb(args.model)

    proba = model.predict_proba(X)[:, 1]

    roc = roc_auc_score(y, proba)
    pr_auc = plot_pr(y, proba, Path(args.out) / "figures" / "pr_curve.png")
    plot_roc(y, proba, Path(args.out) / "figures" / "roc_curve.png")

    # threshold orientado a recall con constraint de precision mínima
    thresholds = np.linspace(0.05, 0.95, 19)
    best = {"thr": 0.5, "recall": 0.0, "precision": 0.0, "f1": 0.0}
    from sklearn.metrics import precision_recall_fscore_support
    for t in thresholds:
        yhat = (proba >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
        if r > best["recall"] and p >= 0.2:
            best = {"thr": float(t), "recall": float(r), "precision": float(p), "f1": float(f1)}

    # matrix & report
    yhat_best = (proba >= best["thr"]).astype(int)
    cm = confusion_matrix(y, yhat_best).tolist()
    report = classification_report(y, yhat_best, output_dict=True)

    out_dir = Path(args.out)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    dump_metrics({
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "best_threshold": best,
        "confusion_matrix": cm,
        "classification_report": report
    }, str(out_dir / "metrics.json"))
    print(f"[OK] Métricas guardadas en {out_dir / 'metrics.json'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="Ruta a test.csv")
    ap.add_argument("--model", required=True, help="Ruta a model_xgb.json")
    ap.add_argument("--scaler", default="models/scaler.pkl", help="Ruta a scaler.pkl (opcional)")
    ap.add_argument("--out", default="reports", help="Carpeta de salida")
    main(ap.parse_args())
