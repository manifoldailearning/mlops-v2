import argparse
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.model import Model

# Parse arguments for model details
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--model-uri', type=str, required=True)
args = parser.parse_args()

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Register the model to SageMaker Model Registry
model = Model(
    image_uri=f"{sagemaker_session.default_bucket()}/model/image-uri:latest",
    model_data=args.model_uri,
    role=role
)

model_package = model.register(
    content_types=["application/json"],
    response_types=["application/json"],
    model_package_group_name=args.model_name,
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"]
)

print(f"Model {args.model_name} registered successfully!")
