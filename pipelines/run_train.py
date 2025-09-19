from sagemaker.xgboost.estimator import XGBoost
import sagemaker
from sagemaker.inputs import TrainingInput

env = 'dev'

role = sagemaker.get_execution_role()

hyperparameters = {
    "num_round": 300,
    "max_depth": 6,
    "eta": 0.2,
    "objective": "binary:logistic",
}

xgb_estimator = XGBoost(
        entry_point='train_xgb.py',
        source_dir='src/training',
        hyperparameters=hyperparameters,
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        framework_version="1.0-1",
        output_path=f"s3://djenk-churn/{env}/models/"
    )

train_s3 = f's3://djenk-churn/{env}/features/train.csv'
val_s3 = f's3://djenk-churn/{env}/features/val.csv'

xgb_estimator.fit({
    "train": TrainingInput(train_s3, content_type='text/csv'),
    "validation": TrainingInput(val_s3, content_type='text/csv')
})
