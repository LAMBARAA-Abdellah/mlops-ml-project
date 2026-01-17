import json
from pathlib import Path
import yaml
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

from src.data import load_dataset
from src.features import build_numeric_preprocess
from src.model import build_model

def load_cfg(path="config/train.yaml"):
    return yaml.safe_load(open(path, "r"))

def save_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    cfg = load_cfg()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    art_dir = PROJECT_ROOT / cfg["artifacts_dir"]
    art_dir.mkdir(exist_ok=True)

    X, y = load_dataset(cfg)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=y
    )

    pipe = Pipeline(steps=[
        ("preprocess", build_numeric_preprocess()),
        ("model", build_model(cfg))
    ])

    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")

    joblib.dump(pipe, art_dir / "model.joblib")
    json.dump(
        {"accuracy": acc, "f1_macro": f1},
        open(art_dir / "metrics.json", "w"),
        indent=2
    )

    save_confusion_matrix(yte, pred, art_dir / "confusion_matrix.png")

    print("TRAIN OK", acc, f1)

if __name__ == "__main__":
    main()
