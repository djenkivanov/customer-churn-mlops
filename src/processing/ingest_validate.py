import argparse
import pandas as pd
from pathlib import Path


def main():
    args = get_args()

    telco_churn = Path(args.input_dir)/"telco-churn.csv"

    # print(telco_churn)

    df = pd.read_csv(telco_churn)
    # df = pd.read_csv("s3://djenk-churn/dev/rawdata/telco-churn.csv")
    # print(df)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(Path(args.output_dir)/"telco-churn.parquet")
   

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to input directory where telco-churn.csv can be found"
        )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory"
        )
    return parser.parse_args()

if __name__ == "__main__":
    main()
