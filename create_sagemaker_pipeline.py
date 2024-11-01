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


# Training Step
training_estimator = Estimator(
    image_uri=training_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.c5.xlarge",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1
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

# Model Evaluation Step
evaluation_script_uri = "s3://sagemaker-us-east-1-866824485776/processed-data/evaluation.py"  # Upload your evaluation script to S3

evaluation_processor = ScriptProcessor(
    image_uri="866824485776.dkr.ecr.us-east-1.amazonaws.com/demo-sagemaker-multimodel-training:latest",  # Using the training image
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
    code=evaluation_script_uri,
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
            right=10.0  # Threshold for the evaluation metric
        )
    ],
    if_steps=[],
    else_steps=[]
)

# Model Registration Step (Conditional on Evaluation)
# Model Registration Step
model = Model(
    image_uri=model_registry_image_uri,
    role=role,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts  # Use training output directly
)

register_step = RegisterModel(
    name="RegisterModelStep",
    model=model,
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="MyModelPackageGroup",
    environment={
        "MODEL_REGISTRY": "true",
        "MODEL_NAME": "MyModel",
        "MODEL_URI": training_step.properties.ModelArtifacts.S3ModelArtifacts
    }
)
# Add Model Registration Step to `if_steps` if condition is met
condition_step.if_steps = [register_step]

# Create the SageMaker Pipeline
pipeline = Pipeline(
    name="MLOpsPipeline",
    steps=[processing_step, training_step, evaluation_step, condition_step],
    sagemaker_session=sagemaker_session
)

# Create or update the pipeline
pipeline.upsert(role_arn=role)

print("Pipeline created successfully!")