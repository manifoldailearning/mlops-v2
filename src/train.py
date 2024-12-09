import os
import pandas as pd
import xgboost as xgb
import logging
from sklearn.model_selection import train_test_split
from joblib import dump

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_training_data():
    """
    Load training data from SageMaker's expected input path.
    """
    data_dir = "/opt/ml/input/data/train"
    data_path = os.path.join(data_dir, "processed_data.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    logger.info("Loading training data.")
    data = pd.read_csv(data_path)
    return data

def split_data(data, target_column):
    """
    Split the dataset into features and target.
    """
    logger.info("Splitting data into features and target.")
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y

def train_model(X, y, hyperparameters):
    """
    Train the XGBoost model using the given hyperparameters.
    """
    logger.info("Training the XGBoost model.")

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(hyperparameters.get("num_round", 100)),
        max_depth=int(hyperparameters.get("max_depth", 6)),
        learning_rate=float(hyperparameters.get("eta", 0.3)),
        random_state=42
    )

    model.fit(X, y)
    return model

def save_model(model):
    """
    Save the trained model in SageMaker's expected output path.
    """
    logger.info("Saving the model.")
    model_dir = "/opt/ml/model"
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, "xgboost_model.joblib")
    dump(model, model_file)
    logger.info(f"Model saved at {model_file}")

if __name__ == "__main__":
    # SageMaker passes hyperparameters as environment variables
    hyperparameters = {
        "num_round": os.getenv("SM_HP_num_round", 100),
        "max_depth": os.getenv("SM_HP_max_depth", 6),
        "eta": os.getenv("SM_HP_eta", 0.3)
    }

    target_column = os.getenv("TARGET_COLUMN", "target")

    try:
        # Load and prepare data
        data = load_training_data()
        X, y = split_data(data, target_column)

        # Train and save the model
        model = train_model(X, y, hyperparameters)
        save_model(model)

        logger.info("Training job completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise
