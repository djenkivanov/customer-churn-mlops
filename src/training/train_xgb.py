import argparse
import os
import xgboost as xgb
import pickle as pkl
import logging
import pandas as pd

def main():
    args = get_args()

    train_hp = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        "eval_metric": ["auc", "aucpr", "logloss"],
        'objective': args.objective
    }

    train_path = os.path.join(args.train, 'train.csv')
    val_path = os.path.join(args.validation, 'val.csv')

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    y_train = train_df.iloc[:, 0].values
    X_train = train_df.iloc[:, 1:].values

    y_val = val_df.iloc[:, 0].values
    X_val = val_df.iloc[:, 1:].values

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    watchlist = [(dtrain, 'train'), (dval, 'validation')]

    bst = xgb.train(
        params=train_hp,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round
    )

    model_location = os.path.join(args.model_dir, 'model.json')
    logging.info(f'Stored trained model at {model_location}')
    # pkl.dump(bst, open(model_location, 'wb'))
    bst.save_model(model_location)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_round', type=int)
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--objective', type=str, default='reg:squarederror')

    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    return parser.parse_args()


if __name__ == "__main__":
    main()