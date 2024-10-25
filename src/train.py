#!/usr/bin/env python3

import sys
from pathlib import Path
import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import optuna
import tarfile
import joblib  # Import joblib to save the model

# Adding the project root directory to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config.core import config
from src.processing.data_management import save_pipeline
from src.pipeline import pipe_transformer  # Importing the pre-built transformer pipeline
from src import __version__ as _version

_logger = logging.getLogger(__name__)

def setup_logging():
    """
    Setup logging configuration.
    """
    logging.basicConfig(
        filename='/opt/ml/output/logs.log',  # Log to the standard SageMaker output directory
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def load_training_data():
    """
    Load the training data from SageMaker's input path.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    # Use SageMaker's default input directory for training data
    data_dir = "/opt/ml/input/data/train"
    data_path = os.path.join(data_dir, "processed_data.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    _logger.info("Loading dataset for training.")
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip().str.lower()
    return data

def split_data(data):
    """
    Split the dataset into training and test sets.

    Args:
        data (pd.DataFrame): The complete dataset.

    Returns:
        Tuple: Split data (X_train, X_test, y_train, y_test).
    """
    _logger.info("Splitting the dataset into training and test sets.")
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.variables.target, axis=1),
        data[config.variables.target],
        test_size=config.training_params.test_size,
        random_state=config.training_params.random_state
    )
    return X_train, X_test, y_train, y_test

def tune_hyperparameters(X_train, y_train, n_trials=10) -> optuna.study.Study:
    """
    Tune hyperparameters using Optuna.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        n_trials (int): Number of trials for hyperparameter tuning.

    Returns:
        optuna.study.Study: The Optuna study object with the results.
    """
    _logger.info("Starting hyperparameter tuning.")

    def objective(trial):
        xgb_n_estimators = trial.suggest_int('n_estimators', 2, 300)
        xgb_max_depth = trial.suggest_int('max_depth', 3, 10)
        xgb_learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)

        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=config.training_params.random_state,
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate
        )

        score = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1).mean()
        return score

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=config.training_params.random_state)
    )
    study.optimize(objective, n_trials=n_trials)

    _logger.info(f"Best hyperparameters: {study.best_params}")
    return study

def train_best_model(X_train, y_train, best_params) -> xgb.XGBRegressor:
    """
    Train the best model with the given hyperparameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        best_params (dict): Best hyperparameters from Optuna.

    Returns:
        xgb.XGBRegressor: The trained XGBoost model.
    """
    _logger.info("Training the best model with the tuned hyperparameters.")
    xgb_model_best = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=config.training_params.random_state,
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate']
    )
    xgb_model_best.fit(X_train, y_train)
    return xgb_model_best

def model_training(n_trials=10) -> optuna.study.Study:
    """
    Train the model with preprocessing and hyperparameter tuning.

    Args:
        n_trials (int): Number of trials for hyperparameter tuning.

    Returns:
        optuna.study.Study: The Optuna study object.
    """
    try:
        # Load the dataset from SageMaker input path
        data = load_training_data()

        # Split the dataset
        X_train, X_test, y_train, y_test = split_data(data)

        # Fit the pipeline transformer
        _logger.info("Fitting the data preprocessing pipeline.")
        pipe_transformer.fit(X_train, y_train)
        X_train = pipe_transformer.transform(X_train)
        X_test = pipe_transformer.transform(X_test)  # Transform the test set as well

        # Hyperparameter tuning
        study = tune_hyperparameters(X_train, y_train, n_trials=n_trials)

        # Train the best model
        xgb_model_best = train_best_model(X_train, y_train, study.best_params)

        # Save the model to the expected output directory
        model_output_dir = "/opt/ml/model"  # This is where SageMaker expects the model to be saved
        os.makedirs(model_output_dir, exist_ok=True)
        model_file_path = os.path.join(model_output_dir, "xgboost_model.joblib")
        joblib.dump(xgb_model_best, model_file_path)

        # Create a tarball of the saved model
        with tarfile.open(os.path.join(model_output_dir, "model.tar.gz"), "w:gz") as tar:
            tar.add(model_file_path, arcname="xgboost_model.joblib")

        # Save the pipeline and study (optional)
        save_pipeline(
            pipeline_to_persist=pipe_transformer,
            study_to_persist=study,
            model_to_persist=xgb_model_best
        )

        _logger.info(f"Training complete for model version: {_version}")
        return study

    except Exception as e:
        _logger.error(f"An error occurred during model training: {e}")
        sys.exit(1)

if __name__ == '__main__':
    setup_logging()
    model_training(n_trials=10)
#test