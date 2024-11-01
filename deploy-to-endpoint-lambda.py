import json
import boto3
import os

sagemaker_client = boto3.client('sagemaker')

def lambda_handler(event, context):
    print("Received event:", json.dumps(event))
    
    # Extract the model package ARN from the event
    model_package_arn = event['detail']['ModelPackageArn']
    
    # Define endpoint configurations
    endpoint_name = "MyApprovedModelEndpoint"
    endpoint_config_name = "MyApprovedModelEndpointConfig"
    
    try:
        # Create an endpoint configuration
        response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_package_arn,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.large',
                    'InitialVariantWeight': 1.0
                },
            ]
        )
        print(f"Created endpoint configuration: {response}")
        
        # Create or update the endpoint
        try:
            # Attempt to update the endpoint if it already exists
            response = sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            print(f"Updated endpoint: {response}")
        except sagemaker_client.exceptions.ResourceNotFound:
            # Endpoint does not exist, create a new one
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
