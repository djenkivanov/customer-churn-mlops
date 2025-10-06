import sagemaker, json, boto3
from sagemaker import image_uris
from sagemaker.model import Model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env = 'dev'
# env = ParameterString(name="Env", default_value="dev")

role = sagemaker.get_execution_role()
region = sagemaker.Session().boto_region_name
image = image_uris.retrieve("xgboost", region=region, version="1.0-1")

s3 = boto3.client("s3")

obj = s3.get_object(Bucket='djenk-churn', Key=f'{env}/models/latest/model_artifact.json')
model_artifact = json.loads(obj["Body"].read())["model_artifact"]

boto_sess = boto3.Session(region_name=region)
sm_sess = sagemaker.Session(boto_session=boto_sess)

model = Model(
    image_uri=image,
    model_data=model_artifact,
    role=role,
    sagemaker_session=sm_sess
)

registration = model.register(
    model_package_group_name="ChurnPrediction",
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    description="XGBoost Churn Model",
    approval_status="PendingManualApproval"
)

logger.info(f'Model package ARN: {registration.model_package_arn}')

