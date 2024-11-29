import os
import tarfile
import boto3
from flask import Flask, jsonify, request

# Initialize Flask app
app = Flask(__name__)

# Define default model path
model_path = os.getenv('MODEL_PATH', '/opt/ml/model')
model_uri = os.getenv('MODEL_URI')  # S3 URI for the model artifact

# Ensure model directory exists
os.makedirs(model_path, exist_ok=True)

# Download and extract model.tar.gz if MODEL_URI is set
if model_uri:
    print(f"Downloading and extracting model from {model_uri}...")
    s3 = boto3.client('s3')
    
    # Parse bucket and key from MODEL_URI
    if model_uri.startswith("s3://"):
        bucket, key = model_uri.replace("s3://", "").split("/", 1)
    else:
        raise ValueError("Invalid MODEL_URI. Expected an S3 URI.")

    # Download model.tar.gz
    local_tar_path = os.path.join(model_path, 'model.tar.gz')
    s3.download_file(bucket, key, local_tar_path)
    
    # Extract model.tar.gz
    with tarfile.open(local_tar_path) as tar:
        tar.extractall(path=model_path)
    print(f"Model extracted to {model_path}")

# Load the model file (e.g., xgboost_model.joblib)
import joblib
model_file = os.path.join(model_path, 'xgboost_model.joblib')
model = None
if os.path.exists(model_file):
    print(f"Loading model from {model_file}...")
    model = joblib.load(model_file)
    print("Model loaded successfully.")
else:
    raise FileNotFoundError(f"Model file not found at {model_file}")

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
    data = request.get_json()
    if not isinstance(data, list):
        return jsonify(error="Invalid input format. Expected a list."), 400

    predictions = model.predict(data)
    return jsonify(predictions.tolist())

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
