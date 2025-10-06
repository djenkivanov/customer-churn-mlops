from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
import sagemaker
# from sagemaker.workflow.parameters import ParameterString


env = "dev"

# csv_source_path = ParameterString(name="RawInputS3Uri", default_value=f"s3://djenk-churn/{env}/raw")
csv_source_path = f"s3://djenk-churn/{env}/raw"

role = sagemaker.get_execution_role()

processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1
)

processor.run(
    code="src/processing/ingest_validate.py",
    inputs=[
        ProcessingInput(
            source=csv_source_path,
            destination="/opt/ml/processing/input/raw"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/processed",
            destination=f"s3://djenk-churn/{env}/processed"
        )
    ],
    arguments=[
        "--input_dir", "/opt/ml/processing/input/raw",
        "--output_dir", "/opt/ml/processing/output/processed"
    ]
)