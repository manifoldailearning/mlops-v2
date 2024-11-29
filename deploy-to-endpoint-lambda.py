import json
import boto3
import os
import time
from datetime import datetime

sagemaker_client = boto3.client('sagemaker')
s3_uri_loacation = os.getenv('S3_URI_LOCATION')

def lambda_handler(event, context):
    print("Received event:", json.dumps(event))
    
    # Extract the model package ARN from the event
    model_package_arn = event.get('detail', {}).get('ModelPackageArn')
    if not model_package_arn:
        raise ValueError("ModelPackageArn not found in event details.")
    
    # Generate unique names for model, endpoint config, and endpoint
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = f"MyApprovedModel-{timestamp}"
    endpoint_name = f"MyApprovedModelEndpoint"
    endpoint_config_name = f"MyApprovedModelEndpointConfig-{timestamp}"

    # Environment variables
    sagemaker_role_arn = os.getenv('SAGEMAKER_ROLE_ARN')
    if not sagemaker_role_arn:
        raise ValueError("Environment variable 'SAGEMAKER_ROLE_ARN' must be set.")

    max_retries = 3
    sleep_seconds = 10  # Seconds to wait between retries
    endpoint_creation_timeout = 600  # Timeout for endpoint creation (seconds)

    try:
        # Step 1: Create a SageMaker Model
        for attempt in range(max_retries):
            try:
                response = sagemaker_client.create_model(
                    ModelName=model_name,
                    PrimaryContainer={
                        'ModelPackageName': model_package_arn,
                        'Environment': {
                            "MODEL_NAME": model_name,
                            "MODEL_URI":s3_uri_loacation
                        },  # Add environment variables if needed
                    },
                    ExecutionRoleArn=sagemaker_role_arn
                )
                print(f"Created model: {response}")
                break
            except sagemaker_client.exceptions.ClientError as e:
                if "already exists" in str(e):
                    print("Model already exists, proceeding with the existing model.")
                    break
                else:
                    if attempt < max_retries - 1:
                        print(f"Retrying model creation in {sleep_seconds} seconds...")
                        time.sleep(sleep_seconds)
                    else:
                        raise

        # Step 2: Create an Endpoint Configuration
        for attempt in range(max_retries):
            try:
                response = sagemaker_client.create_endpoint_config(
                    EndpointConfigName=endpoint_config_name,
                    ProductionVariants=[
                        {
                            'VariantName': 'AllTraffic',
                            'ModelName': model_name,
                            'InitialInstanceCount': 1,
                            'InstanceType': 'ml.m5.large',
                            'InitialVariantWeight': 1.0
                        },
                    ]
                )
                print(f"Created endpoint configuration: {response}")
                break
            except sagemaker_client.exceptions.ClientError as e:
                if "already exists" in str(e):
                    print("Endpoint configuration already exists, proceeding with the existing config.")
                    break
                else:
                    if attempt < max_retries - 1:
                        print(f"Retrying endpoint configuration creation in {sleep_seconds} seconds...")
                        time.sleep(sleep_seconds)
                    else:
                        raise

        # Step 3: Create or Update the Endpoint
        try:
            try:
                response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                print(f"Endpoint exists: {endpoint_name}, updating...")
                sagemaker_client.update_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config_name
                )
            except sagemaker_client.exceptions.ClientError as e:
                if "Could not find endpoint" in str(e):
                    print(f"Endpoint {endpoint_name} does not exist, creating a new one.")
                    response = sagemaker_client.create_endpoint(
                        EndpointName=endpoint_name,
                        EndpointConfigName=endpoint_config_name
                    )
                else:
                    raise

            # Step 4: Wait for Endpoint to Be InService
            print(f"Waiting for endpoint {endpoint_name} to be InService...")
            elapsed_time = 0
            while elapsed_time < endpoint_creation_timeout:
                response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                endpoint_status = response['EndpointStatus']
                print(f"Endpoint status: {endpoint_status}")

                if endpoint_status == 'InService':
                    print(f"Endpoint is now in service: {endpoint_name}")
                    break
                elif endpoint_status == 'Failed':
                    raise Exception(f"Endpoint creation failed for {endpoint_name}")

                time.sleep(sleep_seconds)
                elapsed_time += sleep_seconds

            if elapsed_time >= endpoint_creation_timeout:
                raise TimeoutError(f"Endpoint creation timed out for {endpoint_name}")

        except Exception as e:
            print(f"Error during endpoint creation or update: {str(e)}")
            raise

        return {
            'statusCode': 200,
            'body': json.dumps(f"Deployment started successfully for endpoint: {endpoint_name}")
        }

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error occurred during deployment: {str(e)}")
        }
