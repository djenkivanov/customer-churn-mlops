import argparse
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    args = get_args()

    telco_csv = Path(args.input_dir)/"telco-churn-ingested.csv"

    df = pd.read_csv(telco_csv)

    splits = split_features(df)
    save_splits(splits, Path(args.output_dir))


def split_features(df):
    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"].map({"No": 0, "Yes": 1})
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=1
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=1
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_splits(splits, output_dir):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pd.concat([y_train, X_train], axis=1).to_csv(output_dir / "train.csv", index=False)
    pd.concat([y_val, X_val], axis=1).to_csv(output_dir / "val.csv", index=False)
    pd.concat([y_test, X_test], axis=1).to_csv(output_dir / "test.csv", index=False)

    logger.info(f"Train/val/test splits saved as .csv in {output_dir}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to input directory where telco-churn.parquet can be found"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path for output directory"
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()