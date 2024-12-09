import boto3
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor, ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.xgboost import XGBoost

from pathlib import Path
# Adding the project root directory to sys.path
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config.core import config

# Initialize sessions and role
region = boto3.Session().region_name
sagemaker_session = sagemaker.session.Session()
role = config.role_arn
default_bucket = sagemaker_session.default_bucket()
pipeline_session = PipelineSession()
model_package_group_name = "EndEndPackage"

# Upload train.py to S3
train_script_s3_uri = f"s3://{default_bucket}/scripts/train.py"

# Fetch image URIs from the config
preprocessing_image_uri = config.image_uris['preprocessing']
n_estimators = config.hyperparameters.n_estimators
max_depth =  config.hyperparameters.max_depth
learning_rate = config.hyperparameters.learning_rate



preprocessing_processor = Processor(
    image_uri=preprocessing_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large"
)

processing_step = ProcessingStep(
    name="PreprocessingStep",
    processor=preprocessing_processor,
    inputs=[
        ProcessingInput(
            source=f"s3://{default_bucket}/raw-data/housing_price_dataset.csv",
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{default_bucket}/processed-data",
            output_name="processed_data"
        )
    ]
)

# Training Step
training_estimator = XGBoost(
    entry_point="train.py",
    source_dir=".",
    framework_version="1.5-1",
    instance_type="ml.m5.large",
    instance_count=1,
    output_path=f"s3://{default_bucket}/training-output",
    role=role,
    hyperparameters={
        "objective": "reg:squarederror",
        "num_round": 100,
        "max_depth": 6,
        "eta": 0.3
    },
    environment={
        "TARGET_COLUMN": "price"  # Update as needed
    }
)

training_step = TrainingStep(
    name="TrainingStep",
    estimator=training_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                "processed_data"
            ].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)

# Evaluation Step
# Use SageMaker's built-in XGBoost container for evaluation
evaluation_image_uri = sagemaker.image_uris.retrieve("xgboost", region, "1.5-1")

evaluation_processor = ScriptProcessor(
    image_uri=evaluation_image_uri,
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=sagemaker_session
)

evaluation_output_file = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation_output",
    path="evaluation.json"
)

evaluation_step = ProcessingStep(
    name="ModelEvaluationStep",
    processor=evaluation_processor,
    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=f"s3://{default_bucket}/processed-data/processed_data.csv",
            destination="/opt/ml/processing/test_data"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{default_bucket}/evaluation-output",
            output_name="evaluation_output"
        )
    ],
    code="evaluation.py",
    property_files=[evaluation_output_file]
)

# Condition Step
condition_step = ConditionStep(
    name="CheckModelQuality",
    conditions=[
        ConditionLessThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_output_file,
                json_path="model_quality.mse"
            ),
            right=10.0  # Example MSE threshold
        )
    ],
    if_steps=[],
    else_steps=[]
)

# Model Registration Step
register_step = RegisterModel(
    name="RegisterModelStep",
    estimator=training_estimator,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="MyModelPackageGroup"
)

# Attach model registration to the condition
condition_step.if_steps = [register_step]

# Create the SageMaker Pipeline
pipeline = Pipeline(
    name="MLOpsPipeline",
    steps=[processing_step, training_step, evaluation_step, condition_step],
    sagemaker_session=sagemaker_session
)

# Deploy or update the pipeline
pipeline.upsert(role_arn=role)

print("Pipeline created successfully!")
