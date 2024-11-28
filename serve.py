from flask import Flask, request, jsonify
import joblib
import os
import logging

# Initialize Flask app and logger
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model once the server starts
model = None
model_path = os.getenv('MODEL_PATH', '/opt/ml/model')  # Default path in SageMaker
model_file = os.path.join(model_path, 'model.joblib')

try:
    if os.path.exists(model_file):
        logger.info("Loading model from %s", model_file)
        model = joblib.load(model_file)
        logger.info("Model loaded successfully.")
    else:
        logger.error("Model file not found at %s", model_file)
except Exception as e:
    logger.error("Error loading model: %s", str(e))
    raise

# Health check endpoint
@app.route('/ping', methods=['GET'])
def ping():
    """
    Health check endpoint.
    Returns HTTP 200 if the model is loaded successfully, otherwise HTTP 503.
    """
    health = model is not None
    status = 200 if health else 503
    logger.info("Ping request: Model health = %s", "Healthy" if health else "Unhealthy")
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
    logger.info("Received input data: %s", input_data)

    # Validate input data format
    if input_data is None or not isinstance(input_data, list):
        logger.error("Invalid input data: Expected a list")
        return jsonify({'error': 'Invalid input data: Expected a list'}), 400

    try:
        # Perform prediction
        predictions = model.predict(input_data)
        logger.info("Predictions: %s", predictions.tolist())
        return jsonify(predictions.tolist())
    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask server on the expected host and port
    app.run(host='0.0.0.0', port=8080)
