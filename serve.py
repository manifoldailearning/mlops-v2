from flask import Flask, request, jsonify
import joblib
import os
import tarfile
import logging

# Initialize Flask app and logger
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
model_dir = os.getenv('MODEL_PATH', '/opt/ml/model')
tar_file = os.path.join(model_dir, 'model.tar.gz')
model_file = os.path.join(model_dir, 'xgboost_model.joblib')

# Extract model if tar file exists
if os.path.exists(tar_file):
    try:
        logger.info(f"Extracting model from {tar_file}")
        with tarfile.open(tar_file) as tar:
            tar.extractall(path=model_dir)
        logger.info("Model extracted successfully.")
    except Exception as e:
        logger.error(f"Error extracting model: {str(e)}")
        raise
else:
    logger.error(f"Tar file {tar_file} not found. Ensure the model artifact is uploaded correctly.")

# Load the model
model = None
try:
    if os.path.exists(model_file):
        logger.info(f"Loading model from {model_file}")
        model = joblib.load(model_file)
        logger.info("Model loaded successfully.")
    else:
        logger.error(f"Model file {model_file} not found after extraction.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Health check endpoint
@app.route('/ping', methods=['GET'])
def ping():
    """
    Health check endpoint.
    Returns 200 OK if the model is loaded successfully, otherwise 503.
    """
    health = model is not None
    status = 200 if health else 503
    logger.info(f"Ping request: Model health = {'Healthy' if health else 'Unhealthy'}")
    return jsonify({'status': 'Healthy' if health else 'Unhealthy'}), status

# Inference endpoint
@app.route('/invocations', methods=['POST'])
def predict():
    """
    Model inference endpoint.
    Expects input in JSON format and returns predictions.
    """
    if not request.is_json:
        logger.error("Invalid input format: Expected JSON")
        return jsonify({'error': 'Invalid input format: Expected JSON'}), 400

    input_data = request.get_json()
    logger.info(f"Received input data: {input_data}")

    # Validate input data format
    if input_data is None or not isinstance(input_data, list):
        logger.error("Invalid input data: Expected a list")
        return jsonify({'error': 'Invalid input data: Expected a list'}), 400

    try:
        # Perform prediction
        import numpy as np
        input_array = np.array(input_data)

        if len(input_array.shape) == 1:
            input_array = input_array.reshape(1, -1)  # Convert 1D array to 2D

        logger.info(f"Input array shape: {input_array.shape}")

        predictions = model.predict(input_array)
        logger.info(f"Predictions: {predictions.tolist()}")
        return jsonify(predictions.tolist())
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask server on the expected host and port
    app.run(host='0.0.0.0', port=8080)
