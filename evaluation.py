import json
import pathlib
import pickle
import tarfile
import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    # Load the model from the tar file
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    #the model is saved as 'xgboost-model'
    model = pickle.load(open("xgboost-model", "rb"))

    # Load the test data
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)
    
    # The first column is the target variable (Ex-Showroom_Price)
    y_test = df.iloc[:, 0].to_numpy()
    
    # Drop the target column to get the feature set
    df.drop(df.columns[0], axis=1, inplace=True)
    
    # Convert test data into XGBoost DMatrix format for predictions
    X_test = xgboost.DMatrix(df.values)
    
    # Make predictions
    predictions = model.predict(X_test)

    # Calculate mean squared error and standard deviation
    mse = mean_squared_error(y_test, predictions)
    std = np.std(y_test - predictions)

    # Create the evaluation report
    report_dict = {
        "regression_metrics": {
            "mse": {
                "value": mse,
                "standard_deviation": std
            },
        },
    }

    # Save the evaluation report to the output directory
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
