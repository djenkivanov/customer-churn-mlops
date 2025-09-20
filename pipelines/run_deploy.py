import boto3
from sagemaker.model import ModelPackage
from sagemaker.predictor import Predictor
import sagemaker
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

env = 'dev'

sess = sagemaker.Session()
sm = sess.sagemaker_client

role = sagemaker.get_execution_role()

# delete old endpoint config
# try:
#     sm.delete_endpoint_config(EndpointConfigName='churn-xgb-dev')
# except botocore.exceptions.ClientError:
#     pass

response = sm.list_model_packages(
    ModelPackageGroupName='ChurnPrediction',
    ModelApprovalStatus='Approved',
    SortBy='CreationTime',
    SortOrder='Descending',
    MaxResults=1
)

model_package_arn = response['ModelPackageSummaryList'][0]['ModelPackageArn']

model = ModelPackage(role=role, model_package_arn=model_package_arn)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=f'churn-xgb-{env}',
    update_endpoint=True,
    data_capture_config=sagemaker.model_monitor.DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        destination_s3_uri=f's3://djenk-churn/{env}/endpoint-capture'
    )
)

if predictor is None:
    predictor = Predictor(endpoint_name='churn-xgb-dev', sagemaker_session=sess)

predictor.serializer = CSVSerializer()
predictor.deserializer = JSONDeserializer()

test_predict = "0,0,1,1,6,1,0,19.7,113.5,0,0,0,0,1,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,1"
result = predictor.predict(test_predict)

print(f'Predicted probability of churn: {result}')