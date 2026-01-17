import json
from pathlib import Path
import joblib
import yaml
from sklearn.metrics import classification_report
from src.data import load_dataset

def load_cfg(path="config/train.yaml"):
    return yaml.safe_load(open(path, "r"))

def main():
    cfg = load_cfg()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    art_dir = PROJECT_ROOT / cfg["artifacts_dir"]

    model = joblib.load(art_dir / "model.joblib")

    X, y = load_dataset(cfg)
    pred = model.predict(X)

    report = classification_report(y, pred, output_dict=True)

    json.dump(
        report,
        open(art_dir / "report.json", "w"),
        indent=2
    )

    print("EVALUATE OK")

if __name__ == "__main__":
    main()
