# Trigger lambda when s3 bucket object is updated
import json
import boto3
import os

def lambda_handler(event, context):
    # Initialize SageMaker client
    sagemaker_client = boto3.client('sagemaker')
    
    # Get pipeline name from environment variable
    pipeline_name = os.environ['PIPELINE_NAME']
    
    # Extract bucket name and key from the event
    for record in event['Records']:
        bucket_name = record['s3']['bucket']['name']
        object_key = record['s3']['object']['key']
        
        print(f"New object uploaded to S3 - Bucket: {bucket_name}, Key: {object_key}")
    
    # Start the SageMaker pipeline
    try:
        response = sagemaker_client.start_pipeline_execution(
            PipelineName=pipeline_name
        )
        print(f"Started SageMaker Pipeline Execution. ARN: {response['PipelineExecutionArn']}")
        return {
            'statusCode': 200,
            'body': json.dumps('Pipeline triggered successfully.')
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Failed to trigger pipeline. Error: {str(e)}")
        }
