import boto3
from sagemaker.model import ModelPackage
from sagemaker.predictor import Predictor
import sagemaker
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.model_monitor import DefaultModelMonitor, BaseliningJob, DatasetFormat, ModelQualityMonitor, EndpointInput, CronExpressionGenerator
import pandas as pd
from datetime import datetime
import uuid
from botocore.exceptions import ClientError

def unique_name(base: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{base}-{ts}-{short}"

def endpoint_exists(sm_client, name):
    try:
        sm_client.describe_endpoint(EndpointName=name)
        return True
    except ClientError:
        return False

env = 'dev'
endpoint_name = f'churn-xgb-{env}'

REGION = "eu-north-1" 
sess = sagemaker.Session(boto3.session.Session(region_name=REGION))
sm = sess.sagemaker_client

role = sagemaker.get_execution_role()

response = sm.list_model_packages(
    ModelPackageGroupName='ChurnPrediction',
    ModelApprovalStatus='Approved',
    SortBy='CreationTime',
    SortOrder='Descending',
    MaxResults=1
)

model_package_arn = response['ModelPackageSummaryList'][0]['ModelPackageArn']

model = ModelPackage(role=role, model_package_arn=model_package_arn)

exists = endpoint_exists(sm, endpoint_name)

print(f'exists: {exists}')
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=endpoint_name,
    update_endpoint=exists,
    data_capture_config=sagemaker.model_monitor.DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        capture_options=["REQUEST", "RESPONSE"],
        destination_s3_uri=f's3://djenk-churn/{env}/endpoint-capture'
    )
)

if predictor is None:
    predictor = Predictor(endpoint_name=endpoint_name, sagemaker_session=sess)

predictor.serializer = CSVSerializer()
predictor.deserializer = JSONDeserializer()

test_predict = "0,0,1,1,6,1,0,19.7,113.5,0,0,0,0,1,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,1"
result = predictor.predict(test_predict)

print(f'Predicted probability of churn: {result}')



train_csv = f"s3://djenk-churn/{env}/features/train.csv"
baseline_output = f"s3://djenk-churn/{env}/monitoring/dataquality/baseline"

data_quality = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.t3.large",
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sess
)

data_quality_baseline = data_quality.suggest_baseline(
    baseline_dataset=train_csv,
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=baseline_output,
    wait=True,
)

data_capture_s3 = f's3://djenk-churn/{env}/endpoint-capture'

schedule_dq = data_quality.create_monitoring_schedule(
    monitor_schedule_name=unique_name(f"churn-xgb-{env}-dataquality-daily"),
    output_s3_uri=f's3://djenk-churn/{env}/monitoring/dataquality/reports',
    statistics=data_quality.baseline_statistics(),
    constraints=data_quality.suggested_constraints(),
    schedule_cron_expression=CronExpressionGenerator.daily(hour=0),
    enable_cloudwatch_metrics=True,
    endpoint_input=EndpointInput(
        endpoint_name=f"churn-xgb-{env}",
        destination="/opt/ml/processing/input/endpoint"
    )
)

print("DataQuality schedule created.")





model_quality = ModelQualityMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.t3.large",
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sess,
)

val_csv_s3 = f's3://djenk-churn/{env}/features/val_preds.csv'
model_quality_out_s3 = f's3://djenk-churn/{env}/monitoring/modelquality/baseline'

model_quality_baseline = model_quality.suggest_baseline(
    baseline_dataset=val_csv_s3,
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=model_quality_out_s3,
    problem_type="BinaryClassification",
    inference_attribute='prediction',
    ground_truth_attribute='Churn'
)

model_quality_baseline.wait(logs=False)

mq_baseline_job = model_quality.latest_baselining_job

mq_constraints = pd.DataFrame(mq_baseline_job.suggested_constraints().body_dict["binary_classification_constraints"]).T

print(mq_constraints)
ground_truth_s3 = f's3://djenk-churn/{env}/ground-truth'

schedule_mq = model_quality.create_monitoring_schedule(
    monitor_schedule_name=unique_name(f"churn-xgb-{env}-modelquality-daily"),
    problem_type="BinaryClassification",
    output_s3_uri=f"s3://djenk-churn/{env}/monitoring/modelquality/reports",
    schedule_cron_expression=CronExpressionGenerator.daily(hour=0),
    constraints=model_quality.suggested_constraints(),
    ground_truth_input=ground_truth_s3,
    enable_cloudwatch_metrics=True,
    endpoint_input=EndpointInput(
        endpoint_name=f"churn-xgb-{env}",
        destination="/opt/ml/processing/input/endpoint",
        start_time_offset="-P2D",
        end_time_offset="-P1D",
        inference_attribute="prediction"
    )
)

print("ModelQuality schedule created.")

