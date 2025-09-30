import logging

import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.model_monitor import DatasetFormat
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo, ConditionLessThanOrEqualTo
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.xgboost.estimator import XGBoost

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sess = PipelineSession()
role = sagemaker.get_execution_role()
region = sess.boto_region_name

env    = ParameterString("Env", default_value="dev")
bucket = ParameterString("Bucket", default_value="djenk-churn")
raw    = ParameterString("RawInputS3Uri", default_value="s3://djenk-churn/dev/raw/")

models_base = Join(on="", values=["s3://", bucket, "/", env, "/models"])
telco_csv_uri = Join(on="", values=["s3://", bucket, "/", env, "/processed"])
features_s3_uri = Join(on="", values=["s3://", bucket, "/", env, "/features"])

processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("xgboost", region, version="1.0-1"),
    role=role,
    instance_type="ml.t3.large",
    instance_count=1,
    sagemaker_session=sess,
    command=["python3"],
)

step_ingest = ProcessingStep(
    name="Ingest",
    processor=processor,
    code="src/processing/ingest_validate.py",
    inputs=[
        ProcessingInput(
            source=raw,
            destination="/opt/ml/processing/input/raw"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="ingested",
            source="/opt/ml/processing/output/processed",
            destination=telco_csv_uri
        )
    ],
    job_arguments=[
        "--input_dir", "/opt/ml/processing/input/raw",
        "--output_dir", "/opt/ml/processing/output/processed"
    ]
)

logger.info("Data ingest completed.")

step_split = ProcessingStep(
    name="Split",
    processor=processor,
    code="src/processing/split_features.py",
    inputs=[
        ProcessingInput(
            source=step_ingest.properties.ProcessingOutputConfig.Outputs["ingested"].S3Output.S3Uri,
            destination="/opt/ml/processing/input/processed"
        )
    ],
    outputs=[
        ProcessingOutput(
                output_name="features",
                source="/opt/ml/processing/output/features",
                destination=features_s3_uri
            )
    ],
    job_arguments=[
        "--input_dir", "/opt/ml/processing/input/processed",
        "--output_dir", "/opt/ml/processing/output/features"
    ]
)

logger.info("Split features completed.")

hyperparameters = {
    "num_round": 300,
    "max_depth": 6,
    "eta": 0.2,
    "objective": "binary:logistic",
}

xgb = XGBoost(
    entry_point="train_xgb.py",
    source_dir="src/training",
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="1.0-1",
    output_path=models_base,
    sagemaker_session=sess,
    hyperparameters=hyperparameters,
)

features_uri = step_split.properties.ProcessingOutputConfig.Outputs["features"].S3Output.S3Uri
train_csv = Join(on="", values=[features_uri, "/train.csv"])
val_csv = Join(on="", values=[features_uri, "/val.csv"])
val_preds_s3 = Join(on="", values=[features_s3_uri, "/val_preds.csv"])

step_train = TrainingStep(
    name="Train",
    estimator=xgb,
    inputs={
        "train": TrainingInput(train_csv, content_type="text/csv"),
        "validation": TrainingInput(val_csv, content_type="text/csv")
    }
)

logger.info("Model training completed.")

model_artifact = step_train.properties.ModelArtifacts.S3ModelArtifacts

eval_s3_uri = Join(on="", values=["s3://", bucket, "/", env, "/models/evaluation"])

prop_metrics = PropertyFile(
    name="EvalMetrics",
    output_name="evaluation",
    path="model_metrics.json"
)

step_eval = ProcessingStep(
    name="Evaluate",
    processor=processor,
    code="src/evaluating/evaluate.py",
    inputs=[
        ProcessingInput(
                source=model_artifact,
                destination="/opt/ml/processing/input/model"
            ),
        ProcessingInput(
                source=features_uri,
                destination="/opt/ml/processing/input/data"
            )
    ],
    outputs=[
        ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output/evaluation",
                destination=eval_s3_uri
            )
    ],
    job_arguments=[
        "--model_dir", "/opt/ml/processing/input/model",
        "--data_dir", "/opt/ml/processing/input/data",
        "--output_dir", "/opt/ml/processing/output/evaluation"
    ],
    property_files=[prop_metrics]
)

logger.info("Model evaluation completed.")

evaluation_uri = step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri

cond_auc = ConditionGreaterThanOrEqualTo(
    left=JsonGet(step_name=step_eval.name, property_file=prop_metrics, json_path="roc_auc"),
    right=0.79,
)
cond_prauc = ConditionGreaterThanOrEqualTo(
    left=JsonGet(step_name=step_eval.name, property_file=prop_metrics, json_path="pr_auc"),
    right=0.50,
)
cond_logloss = ConditionLessThanOrEqualTo(
    left=JsonGet(step_name=step_eval.name, property_file=prop_metrics, json_path="logloss"),
    right=0.56,
)
cond_f1 = ConditionGreaterThanOrEqualTo(
    left=JsonGet(step_name=step_eval.name, property_file=prop_metrics, json_path="f1_at_0.5"),
    right=0.50,
)

gate_indicator = ConditionStep(
    name="GateIndicator",
    conditions=[cond_auc, cond_prauc, cond_logloss, cond_f1],
    if_steps=[],
    else_steps=[],
)

metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=Join(on="", values=[evaluation_uri, "/model_metrics.json"]),
        content_type="application/json"
    )
)

step_register = RegisterModel(
    name="RegisterModel",
    estimator=xgb,
    model_data=model_artifact,
    content_types=["text/csv"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="ChurnPrediction",
    model_metrics=metrics,
    approval_status="PendingManualApproval",
)

logger.info("Model register completed.")
endpoint_name = Join(on="", values=["churn-xgb-", env])

check_job_cfg = CheckJobConfig(
    role=role,
    instance_count=1,
    instance_type="ml.t3.large",
    volume_size_in_gb=20,
    sagemaker_session=sess,
    max_runtime_in_seconds=3600,
)

dq_s3 = Join(on="", values=["s3://", bucket, "/", env, "/monitoring/dataquality/baseline/"])

dq_cfg = DataQualityCheckConfig(
    baseline_dataset=train_csv,
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=dq_s3,
)

dq_con = Join(on="", values=[dq_s3, "constraints.json"])
dq_stat = Join(on="", values=[dq_s3, "statistics.json"])

step_dq_baseline = QualityCheckStep(
    name="DataQualityBaseline",
    check_job_config=check_job_cfg,
    quality_check_config=dq_cfg,
    skip_check=True,
    register_new_baseline=True
)

step_dq_check = QualityCheckStep(
    name="DataQualityCheck",
    check_job_config=check_job_cfg,
    quality_check_config=dq_cfg,
    skip_check=False,
    register_new_baseline=False,
    supplied_baseline_statistics=dq_stat,
    supplied_baseline_constraints=dq_con
)

mq_s3 = Join(on="", values=["s3://", bucket, "/", env, "/monitoring/modelquality/baseline/"])

mq_cfg = ModelQualityCheckConfig(
    problem_type="BinaryClassification",
    baseline_dataset=val_preds_s3,
    dataset_format=DatasetFormat.csv(header=True),
    inference_attribute="prediction",
    ground_truth_attribute="Churn",
    output_s3_uri=mq_s3,
)

mq_con = Join(on="", values=[mq_s3, "constraints.json"])
mq_stat = Join(on="", values=[mq_s3, "statistics.json"])

step_mq_baseline = QualityCheckStep(
    name="ModelQualityBaseline",
    check_job_config=check_job_cfg,
    quality_check_config=mq_cfg,
    skip_check=True,
    register_new_baseline=True
)

step_mq_check = QualityCheckStep(
    name="ModelQualityCheck",
    check_job_config=check_job_cfg,
    quality_check_config=mq_cfg,
    skip_check=False,
    register_new_baseline=False,
    supplied_baseline_statistics=mq_stat,
    supplied_baseline_constraints=mq_con
)

deploy_model = Model(
    image_uri=xgb.image_uri,
    model_data=model_artifact,
    role=role,
    sagemaker_session=sess,
    name="something"
)

step_create_model = ModelStep(
    name="CreateInferenceModel",
    step_args=deploy_model.create()
)

pipeline =  Pipeline(
    name="ChurnPredictMLOpsPipeline",
    parameters=[env, bucket, raw],
    steps=[step_ingest, step_split, step_train, step_eval, step_register, step_dq_baseline,
           step_mq_baseline, step_dq_check, step_mq_check],
    sagemaker_session=sess,
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)

    execution = pipeline.start(parameters={
        "Env": "dev",
        "Bucket": "djenk-churn",
        "RawInputS3Uri": "s3://djenk-churn/dev/raw/"
    })

    execution.wait()

    logger.info("Status:", execution.describe()["PipelineExecutionStatus"])