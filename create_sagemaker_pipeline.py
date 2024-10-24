import boto3
import sagemaker
import sagemaker.session
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline
# Adding the project root directory to sys.path
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config.core import config


region = boto3.Session().region_name
sagemaker_session = sagemaker.session.Session() 
role = sagemaker.get_execution_role()
default_bucket = sagemaker_session.default_bucket()
pipeline_session = PipelineSession()
model_package_group_name = "EndEndPackage"

# Fetch image URIs from the config
preprocessing_image_uri = config.image_uris.preprocessing
training_image_uri = config.image_uris.training
model_registry_image_uri = config.image_uris.model_registry

# Preprocessing Step
preprocessing_processor = Processor(
    image_uri= preprocessing_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large"
)

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
    model_package_group_name="MyModelPackageGroup"
)
# Create the SageMaker Pipeline
pipeline = Pipeline(
    name="MLOpsPipeline",
    steps=[processing_step, training_step, register_step],
    sagemaker_session=sagemaker_session
)

# Create or update the pipeline
pipeline.upsert(role_arn=role)

print("Pipeline created successfully!")