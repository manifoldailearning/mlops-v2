import os
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.model import Model

# Fetch arguments from environment variables
model_name = os.getenv('MODEL_NAME')
model_uri = os.getenv('MODEL_URI')

if not model_name or not model_uri:
    raise ValueError("Both MODEL_NAME and MODEL_URI environment variables must be set.")

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Register the model to SageMaker Model Registry
model = Model(
    image_uri=f"{sagemaker_session.default_bucket()}/model/image-uri:latest",
    model_data=model_uri,
    role=role
)

model_package = model.register(
    content_types=["application/json"],
    response_types=["application/json"],
    model_package_group_name=model_name,
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"]
)

print(f"Model {model_name} registered successfully!")
