import os
import tarfile
import boto3
import logging
from flask import Flask, jsonify, request
import joblib
import time
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Define default model path and S3 URI
model_path = os.getenv('MODEL_PATH', '/opt/ml/model')
model_uri = os.getenv('MODEL_URI')  # S3 URI for the model artifact

# Ensure the model directory exists
os.makedirs(model_path, exist_ok=True)

# Retry mechanism for S3 downloads
def download_with_retries(s3_client, bucket, key, local_path, max_retries=3, sleep_time=5):
    """Download a file from S3 with retries and validate the file size."""
    for attempt in range(max_retries):
        try:
            s3_client.download_file(bucket, key, local_path)
            # Log the file size
            file_size = os.path.getsize(local_path)
            logger.info(f"Downloaded file size: {file_size} bytes")
            if file_size == 0:
                raise ValueError("Downloaded file is empty.")
            return  # Download successful
        except (ClientError, ValueError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"Download failed (attempt {attempt + 1}/{max_retries}): {str(e)}. Retrying...")
                time.sleep(sleep_time)
            else:
                logger.error("Max retries reached. Download failed.")
                raise

# Validate and extract the tar.gz file
def validate_and_extract_tar(tar_path, extract_path):
    """Validate and extract a tar.gz file."""
    if tarfile.is_tarfile(tar_path):
        logger.info(f"Validating tar file {tar_path}...")
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                logger.info(f"Extracting: {member.name}")
            tar.extractall(path=extract_path)
        logger.info(f"Model extracted to {extract_path}")
    else:
        raise ValueError(f"The file {tar_path} is not a valid tar.gz archive.")

# Find the xgboost_model.joblib file dynamically after extraction
def find_model_file(extract_path, filename="xgboost_model.joblib"):
    """Search for the model file recursively within the extracted path."""
    for root, dirs, files in os.walk(extract_path):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"Model file '{filename}' not found in {extract_path}")

# Download and extract model.tar.gz if MODEL_URI is set
if model_uri:
    try:
        logger.info(f"Downloading and extracting model from {model_uri}...")
        s3 = boto3.client('s3')

        # Parse bucket and key from MODEL_URI
        if model_uri.startswith("s3://"):
            bucket, key = model_uri.replace("s3://", "").split("/", 1)
        else:
            raise ValueError("Invalid MODEL_URI. Expected an S3 URI.")

        # Download model.tar.gz
        local_tar_path = os.path.join(model_path, 'model.tar.gz')
        download_with_retries(s3, bucket, key, local_tar_path)

        # Validate and extract
        validate_and_extract_tar(local_tar_path, model_path)
    except Exception as e:
        logger.error(f"Error downloading or extracting model: {str(e)}")
        raise
else:
    logger.warning("MODEL_URI is not provided. Assuming the model is already present.")

# Load the model file (e.g., xgboost_model.joblib)
model = None
try:
    # Search for the model file dynamically
    model_file = find_model_file(model_path, "xgboost_model.joblib")
    logger.info(f"Loading model from {model_file}...")
    model = joblib.load(model_file)
    logger.info("Model loaded successfully.")
except FileNotFoundError as e:
    logger.error(str(e))
    raise

# Health check endpoint
@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint."""
    health = model is not None
    status = 200 if health else 503
    return jsonify(status='Healthy' if health else 'Unhealthy'), status

# Prediction endpoint
@app.route('/invocations', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        data = request.get_json()
        if not isinstance(data, list):
            logger.warning("Invalid input format. Expected a list.")
            return jsonify(error="Invalid input format. Expected a list."), 400

        # Perform predictions
        predictions = model.predict(data)
        logger.info(f"Predictions: {predictions}")
        return jsonify(predictions.tolist())
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify(error=f"Prediction failed: {str(e)}"), 500

# Run the Flask app
if __name__ == '__main__':
    try:
        logger.info("Starting the Flask server...")
        app.run(host='0.0.0.0', port=8080)
    except Exception as e:
        logger.error(f"Failed to start Flask server: {str(e)}")
        raise
