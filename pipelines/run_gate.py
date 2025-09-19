import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

env = 'dev'

role = sagemaker.get_execution_role()

processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1
)

processor.run(
    code="src/evaluating/gate_pass.py",
    inputs=[
        ProcessingInput(
            source=f"s3://djenk-churn/{env}/models/evaluation/",
            destination="/opt/ml/processing/input/metrics"
        )
    ],
    arguments=[
        "--metrics_dir", "/opt/ml/processing/input/metrics",
    ]
)