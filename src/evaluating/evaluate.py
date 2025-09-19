import tarfile
import json
from pathlib import Path
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, log_loss, f1_score, precision_score, recall_score, confusion_matrix
import logging
import argparse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    args = get_args()

    model_tar = Path(args.model_dir)/'model.tar.gz'
    out_dir = Path(args.model_dir)/'unzipped'

    with tarfile.open(model_tar) as tar:
        tar.extractall(path=out_dir)

    model_file = next(out_dir.rglob('model.json'))

    booster = xgb.Booster()
    booster.load_model(model_file)

    test_path = Path(args.data_dir) / 'test.csv'
    df = pd.read_csv(test_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values

    dtest = xgb.DMatrix(X)

    prob = booster.predict(dtest)
    preds = (prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "logloss": float(log_loss(y, prob, eps=1e-15)),
        "f1_at_0.5": float(f1_score(y, preds)),
        "precision_at_0.5": float(precision_score(y, preds)),
        "recall_at_0.5": float(recall_score(y, preds)),
        "positive_rate": float(prob.mean()),
    }
    log.info(f"metrics: {metrics}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'model_metrics.json').write_text(json.dumps(metrics, indent=2))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str
    )
    parser.add_argument(
        "--data_dir",
        type=str
    )
    parser.add_argument(
        "--output_dir",
        type=str
    )
    return parser.parse_args()

if __name__ == '__main__':
    main()