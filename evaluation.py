import os
import json
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
import tarfile

# Define the paths
model_path = "/opt/ml/processing/model/model.tar.gz"
test_data_path = "/opt/ml/processing/test_data/processed_data.csv"
evaluation_output_path = "/opt/ml/processing/output/evaluation.json"

# Load the test data
print("Loading test data from:", test_data_path)
test_data = pd.read_csv(test_data_path)

# Assuming 'price' is the target column
target_column = 'price'
X_test = test_data.drop(columns=[target_column])
y_test = test_data[target_column]

# Load the model
print("Loading the model from:", model_path)

with tarfile.open(model_path) as tar:
    tar.extractall(path=".")
    model = joblib.load("xgboost_model.joblib")

# Make predictions
print("Making predictions on the test set...")
predictions = model.predict(X_test)

# Evaluate the model
print("Calculating Mean Squared Error (MSE)...")
mse = mean_squared_error(y_test, predictions)

# Save the evaluation results
print("Saving evaluation results to:", evaluation_output_path)
evaluation_results = {
    "model_quality": {
        "mse": mse
    }
}

# Write the results to the output JSON file
os.makedirs(os.path.dirname(evaluation_output_path), exist_ok=True)
with open(evaluation_output_path, "w") as f:
    json.dump(evaluation_results, f)

print("Evaluation complete.")
