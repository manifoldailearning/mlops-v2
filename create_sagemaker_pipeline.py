import sagemaker
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import RegisterModel
from sagemaker.processing import ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.workflow.pipeline import Pipeline

# Define SageMaker role, bucket, and ECR URIs
role = "<your_sagemaker_role>"
s3_bucket = "<your_s3_bucket>"
ecr_repository = "<your_ecr_repository_uri>"

sagemaker_session = sagemaker.Session()

# Preprocessing Step
preprocessing_processor = ScriptProcessor(
    image_uri=f"{ecr_repository}:preprocessing-latest",
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    command=["python3"]
)
processing_step = ProcessingStep(
    name="PreprocessingStep",
    processor=preprocessing_processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=f"s3://{s3_bucket}/raw-data",
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{s3_bucket}/processed-data"
        )
    ],
    job_arguments=[]
)

# Training Step
training_estimator = Estimator(
    image_uri=f"{ecr_repository}:training-latest",
    role=role,
    instance_count=1,
    instance_type="ml.c5.xlarge",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1
    },
    output_path=f"s3://{s3_bucket}/output",
    sagemaker_session=sagemaker_session
)
training_step = TrainingStep(
    name="TrainingStep",
    estimator=training_estimator,
    inputs={
        "train": sagemaker.TrainingInput(
            s3_data=f"s3://{s3_bucket}/processed-data",
            content_type="text/csv"
        )
    }
)

# Model Registration Step
model = Model(
    image_uri=f"{ecr_repository}:model-registry-latest",
    role=role,
    model_data=f"s3://{s3_bucket}/output/model.tar.gz"
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