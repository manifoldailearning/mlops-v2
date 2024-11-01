from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load model once the server starts
model = None
model_path = os.getenv('MODEL_PATH', '/opt/ml/model')
model_file = os.path.join(model_path, 'model.joblib')
if os.path.exists(model_file):
    model = joblib.load(model_file)

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    health = model is not None
    status = 200 if health else 404
    return '', status

@app.route('/invocations', methods=['POST'])
def predict():
    """Model inference endpoint"""
    if not request.is_json:
        return jsonify({'error': 'Invalid input format'}), 400

    input_data = request.get_json()
    # Assuming input_data is in the form required by your model
    prediction = model.predict(input_data)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    # Run the Flask server on the expected port
    app.run(host='0.0.0.0', port=8080)
