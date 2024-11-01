import json
import boto3
import os
import time
from datetime import datetime

sagemaker_client = boto3.client('sagemaker')

def lambda_handler(event, context):
    print("Received event:", json.dumps(event))
    
    # Extract the model package ARN from the event
    model_package_arn = event['detail']['ModelPackageArn']
    
    # Append timestamp to make names unique
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = f"MyApprovedModel-{timestamp}"
    endpoint_name = f"MyApprovedModelEndpoint-{timestamp}"
    endpoint_config_name = f"MyApprovedModelEndpointConfig-{timestamp}"
    
    max_retries = 3
    sleep_seconds = 10  # Seconds to wait between retries
    endpoint_creation_timeout = 600  # Timeout for endpoint creation (seconds)

    try:
        # Create a SageMaker Model from the model package
        for attempt in range(max_retries):
            try:
                response = sagemaker_client.create_model(
                    ModelName=model_name,
                    PrimaryContainer={
                        'ModelPackageName': model_package_arn,
                        'Environment': {},  # You can add any additional environment variables here
                    },
                    ExecutionRoleArn=os.getenv('SAGEMAKER_ROLE_ARN')  # Set this in Lambda environment variables
                )
                print(f"Created model: {response}")
                break
            except sagemaker_client.exceptions.ClientError as e:
                # If the model already exists, continue to use it
                if e.response['Error']['Code'] == 'ValidationException' and 'already exists' in e.response['Error']['Message']:
                    print("Model already exists, proceeding with the existing model.")
                    break
                else:
                    if attempt < max_retries - 1:
                        print(f"Retrying model creation in {sleep_seconds} seconds...")
                        time.sleep(sleep_seconds)
                    else:
                        raise

        # Create an endpoint configuration
        for attempt in range(max_retries):
            try:
                response = sagemaker_client.create_endpoint_config(
                    EndpointConfigName=endpoint_config_name,
                    ProductionVariants=[
                        {
                            'VariantName': 'AllTraffic',
                            'ModelName': model_name,  # Use the created model name here
                            'InitialInstanceCount': 1,
                            'InstanceType': 'ml.m5.large',
                            'InitialVariantWeight': 1.0
                        },
                    ]
                )
                print(f"Created endpoint configuration: {response}")
                break
            except sagemaker_client.exceptions.ClientError as e:
                # If endpoint config already exists, continue to use it
                if e.response['Error']['Code'] == 'ValidationException' and 'already exists' in e.response['Error']['Message']:
                    print("Endpoint configuration already exists, proceeding with the existing config.")
                    break
                else:
                    if attempt < max_retries - 1:
                        print(f"Retrying endpoint configuration creation in {sleep_seconds} seconds...")
                        time.sleep(sleep_seconds)
                    else:
                        raise

        # Create the endpoint
        try:
            response = sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            print(f"Created endpoint: {response}")

            # Wait for endpoint creation
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

        except sagemaker_client.exceptions.ResourceNotFound:
            print(f"Endpoint not found, retrying creation in {sleep_seconds} seconds...")
            time.sleep(sleep_seconds)
            response = sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            print(f"Created endpoint: {response}")

        return {
            'statusCode': 200,
            'body': json.dumps('Deployment triggered successfully.')
        }

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error occurred during deployment: {str(e)}')
        }
