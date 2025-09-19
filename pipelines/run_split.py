from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
import sagemaker

env = "dev"
# env = ParameterString(name="Env", default_value="dev")

role = sagemaker.get_execution_role()

processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1
)

processor.run(
    code="src/processing/split_features.py",
    inputs=[
        ProcessingInput(
            source=f"s3://djenk-churn/{env}/processed",
            destination="/opt/ml/processing/input/processed"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/split-features",
            destination=f"s3://djenk-churn/{env}/features"
        )
    ],
    arguments=[
        "--input_dir", "/opt/ml/processing/input/processed",
        "--output_dir", "/opt/ml/processing/output/split-features"
    ]
)