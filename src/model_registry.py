import os
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.model import Model

def get_aws_account_id():
    sts = boto3.client('sts')
    return sts.get_caller_identity()["Account"]

def get_aws_region():
    session = boto3.session.Session()
    return session.region_name


# Fetch arguments from environment variables
model_name = os.getenv('MODEL_NAME')
model_uri = os.getenv('MODEL_URI')
account = get_aws_account_id()
region = get_aws_region()

if not model_name or not model_uri:
    raise ValueError("Both MODEL_NAME and MODEL_URI environment variables must be set.")

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Define ECR image URI dynamically
image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/demo-sagemaker-multimodel-model-registry:latest"

# Validate the model URI
if not model_uri.startswith("s3://"):
    raise ValueError(f"Invalid model_uri: {model_uri}. It must be a valid S3 URI.")

# Register the model with SageMaker Model Registry
print(f"Registering model {model_name} with URI {model_uri}...")
model = Model(
    image_uri=image_uri,
    model_data=model_uri,
    role=role,
    sagemaker_session=sagemaker.Session()
)

model_package = model.register(
    content_types=["application/json"],
    response_types=["application/json"],
    model_package_group_name=model_name,
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
)

# Wait for model registration to complete
model_package.wait()
print(f"Model {model_name} registered successfully!")