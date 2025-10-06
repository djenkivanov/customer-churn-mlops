import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    args = get_args()

    metrics_json = Path(args.metrics_dir) / 'model_metrics.json'

    gate_pass(metrics_json)


def gate_pass(model_json):
    with open(model_json) as f:
        metrics = json.load(f)

    gates = {
        "roc_auc": 0.79,
        "pr_auc": 0.5,
        "logloss": 0.56,
        "f1": 0.5,
    }

    failed = {}

    for metric, min_val in gates.items():
        val = metrics.get(metric)
        if metric == 'logloss':
            if val > min_val:
                failed[metric] = val
        else:
            if val < min_val:
                failed[metric] = val

    if failed:
        logger.info(f'Gate pass failed for model, failing metrics: {failed}')
    else:
        logger.info('Gate pass success.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_dir', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    main()