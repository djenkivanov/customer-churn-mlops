import boto3

endpoint = "churn-xgb-dev"
runtime = boto3.client("sagemaker-runtime")

# low churn prob
# payload = '0,1,0,24,1,1,75.35,1808.40,1,0,1,1,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0'

# high churn prob
payload = '0,0,0,2,1,1,95.60,191.20,1,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,1'

response = runtime.invoke_endpoint(
    EndpointName=endpoint,
    ContentType="text/csv",
    Body=payload
)

pred = response["Body"].read().decode('utf-8')

print(pred)