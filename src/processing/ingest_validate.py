import argparse
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    args = get_args()

    telco_churn = Path(args.input_dir)/"telco-churn.csv"

    df = pd.read_csv(telco_churn)

    validate(df)
    val_clean_df = clean(df)

    # fetch from local mounted in processing job
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    val_clean_df.to_parquet(Path(args.output_dir)/"telco-churn.parquet")

    # hardcoded fetch for dev
    # df = pd.read_csv("s3://djenk-churn/dev/raw/telco-churn.csv")

    
    # print(f'df AFTER ========== \n {val_clean_df}')
   

def validate(df):
    domain_checks(df)
    validate_cols(df)
    na_cols(df)
    logger.info("Validation complete.")


def clean(df_param):
    df = clean_whitespaces(df_param)
    df = clean_floats(df)
    df = encode_binary(df)
    df = one_hot_encode(df)
    logger.info("Data cleaning complete.")
    return df

def encode_binary(df):
    binary_cols = [
        "Partner", "Dependents", "PaperlessBilling", "PhoneService"
    ]

    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    logger.info("Binary encoded 'Yes' and 'No' in dataset.")
    return df


def one_hot_encode(df):
    cat_cols = [
        "gender", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    #change drop_first to false in model not linear, preserves structure
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    logger.info("One-hot encoded categorical columns.")
    return df


def clean_whitespaces(df):
    obj_cols = df.select_dtypes(include='object').columns
    df[obj_cols] = df[obj_cols].apply(lambda x: x.str.strip())
    logger.info("Stripped whitespaces from object columns.")
    return df


def clean_floats(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    coerced_na = df['TotalCharges'].isna().sum()
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0).astype('float32')
    
    num_cols = ["MonthlyCharges", "TotalCharges"]
    df[num_cols] = df[num_cols].astype("float32")

    logger.info(f'Coerced rows fixed in column TotalCharges: {coerced_na}')

    return df


def validate_cols(df):
    cols = 21
    if df.shape[1] != 21:
        logger.warning(f"Expected 21 columns but got {df.shape[1]}")
    else:
        logger.info("No unexpected columns found.")


def na_cols(df):
    column_na_counts = df.isna().sum()
    columns_with_na = {}
    for col, na_count in column_na_counts.items():
        if na_count != 0:
            columns_with_na[col] = na_count
    
    if bool(columns_with_na):
        logger.warning(f"Columns with NA values found: {columns_with_na}")
    else:
        logger.info("No NA values detected.")


def domain_checks(df):
    rules = {
        "gender": {"Male", "Female"},
        "Partner": {"Yes", "No"},
        "Dependents": {"Yes", "No"},
        "PhoneService": {"Yes", "No"},
        "MultipleLines": {"Yes", "No", "No phone service"},
        "InternetService": {"DSL", "Fiber optic", "No"},
        "OnlineSecurity": {"Yes", "No", "No internet service"},
        "OnlineBackup": {"Yes", "No", "No internet service"},
        "DeviceProtection": {"Yes", "No", "No internet service"},
        "TechSupport": {"Yes", "No", "No internet service"},
        "StreamingTV": {"Yes", "No", "No internet service"},
        "StreamingMovies": {"Yes", "No", "No internet service"},
        "Contract": {"Month-to-month", "One year", "Two year"},
        "PaperlessBilling": {"Yes", "No"},
        "PaymentMethod": {
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        },
        "Churn": {"Yes", "No"},
    }
    
    for col, allowed in rules.items():
        invalid = set(df[col].unique()) - allowed
        if invalid:
            logger.warning(f"Column {col} has invalid values: {invalid}")

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
        help="Path for output directory"
        )
    return parser.parse_args()

if __name__ == "__main__":
    main()
