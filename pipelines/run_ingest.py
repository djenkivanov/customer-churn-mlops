from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
import sagemaker
# from sagemaker.workflow.parameters import ParameterString


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
    code="src/processing/ingest_validate.py",
    inputs=[
        ProcessingInput(
            source=f"s3://djenk-churn/{env}/raw/",
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