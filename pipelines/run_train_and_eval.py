from sagemaker.xgboost.estimator import XGBoost
import sagemaker, boto3, json
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.processing import ScriptProcessor


env = 'dev'
# env = ParameterString(name="Env", default_value="dev")

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

s3 = boto3.client("s3")
model_artifact = xgb_estimator.model_data
artifact_json = {"model_artifact": model_artifact}

bucket = 'djenk-churn'
key = f'{env}/models/latest/model_artifact.json'

s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(artifact_json).encode("utf-8"))

# harded artifact to skip training for testing
# model_artifact = 's3://djenk-churn/dev/models/sagemaker-xgboost-2025-09-19-15-37-05-461/output/model.tar.gz'

processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("xgboost", sagemaker.Session().boto_region_name, version="1.0-1"),
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    command=['python3']
)

processor.run(
    code='src/evaluating/evaluate.py',
    inputs=[
        ProcessingInput(
            source=model_artifact,
            destination='/opt/ml/processing/input/model'
        ),
        ProcessingInput(
            source=f's3://djenk-churn/{env}/features',
            destination='/opt/ml/processing/input/data'
        )
    ],
    outputs=[
        ProcessingOutput(
            source='/opt/ml/processing/output/evaluation',
            destination=f's3://djenk-churn/{env}/models/evaluation'
        )
    ],
    arguments=[
        '--model_dir', '/opt/ml/processing/input/model',
        '--data_dir', '/opt/ml/processing/input/data',
        '--output_dir', '/opt/ml/processing/output/evaluation'
    ]
)