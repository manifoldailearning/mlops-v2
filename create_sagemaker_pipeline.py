import boto3
import sagemaker
import sagemaker.session
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor,ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep

from pathlib import Path
# Adding the project root directory to sys.path
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config.core import config


region = boto3.Session().region_name
sagemaker_session = sagemaker.session.Session() 
role = config.role_arn
default_bucket = sagemaker_session.default_bucket()
pipeline_session = PipelineSession()
model_package_group_name = "EndEndPackage"

# Fetch image URIs from the config
preprocessing_image_uri = config.image_uris['preprocessing']
training_image_uri = config.image_uris['training']
model_registry_image_uri = config.image_uris['model_registry']
n_estimators = config.hyperparameters.n_estimators
max_depth =  config.hyperparameters.max_depth
learning_rate = config.hyperparameters.learning_rate

# Preprocessing Step
preprocessing_processor = Processor(
    image_uri= preprocessing_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large"
)

# Ensure that dataset is present in the mentioned Source folder

processing_step = ProcessingStep(
    name="PreprocessingStep",
    processor=preprocessing_processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=f"s3://{default_bucket}/raw-data",
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{default_bucket}/processed-data",
            output_name="processed_data" 
        )
    ]
)
training_estimator = Estimator(
    image_uri=sagemaker.image_uris.retrieve("xgboost", "us-east-1", "1.5-1"),
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    source_dir="https://github.com/manifoldailearning/mlops-v2.git",
    entry_point="train.py",
    hyperparameters={
        "objective": "reg:squarederror",
        "num_round": 100,
        "max_depth": 6,
        "eta": 0.3
    },
    environment={
        "TARGET_COLUMN": "price"  # Update as needed
    },
    output_path=f"s3://{default_bucket}/training-output",
    sagemaker_session=sagemaker_session
)

# Define the Training Step
training_step = TrainingStep(
    name="TrainingStep",
    estimator=training_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["processed_data"].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)

# Ensure training_step runs after processing_step
training_step.add_depends_on([processing_step])

evaluation_image_uri = sagemaker.image_uris.retrieve(
    framework="sklearn",
    region=region,
    version="0.23-1"
)

# Updated Evaluation Processor
evaluation_processor = ScriptProcessor(
    image_uri=evaluation_image_uri,
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=sagemaker_session
)

# Property File to Capture Evaluation Results
evaluation_output_file = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation_output",
    path="evaluation.json"
)

# Evaluation Step with Updated Parameters
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
    code="evaluation.py",  # Make sure this file is in the correct path
    property_files=[evaluation_output_file]
)

# Condition Step to Check Model Quality
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
# Model Registration Step (Conditional on Evaluation)

from sagemaker.workflow.step_collections import RegisterModel

# Register Model Step (No environment parameter)
register_step = RegisterModel(
    name="RegisterModelStep",
    estimator=training_estimator,  # Use the training estimator directly
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,  # Use training output
    content_types=["text/csv"],  # Supported input format
    response_types=["text/csv"],  # Supported output format
    inference_instances=["ml.m5.large"],  # Endpoint instance types
    transform_instances=["ml.m5.large"],  # Batch transform instance types
    model_package_group_name="MyModelPackageGroup"
)

# Add the Model Registration Step to the conditional check
condition_step.if_steps = [register_step]

# Create or update the SageMaker Pipeline
pipeline = Pipeline(
    name="MLOpsPipeline",
    steps=[processing_step, training_step, evaluation_step, condition_step],
    sagemaker_session=sagemaker_session
)

# Deploy or update the pipeline in SageMaker
pipeline.upsert(role_arn=role)

print("Pipeline created successfully!")
