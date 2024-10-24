# Module to manage datasets

import logging
import typing as t
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
import optuna
import xgboost as xgb
import os
import tarfile
from pathlib import Path

from src.config.core import config, dataset_path, trained_model_dir
from src import __version__ as _version

_logger = logging.getLogger(__name__)


def load_data(file_name: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Args:
        file_name (str): The name of the CSV file to load.

    Returns:
        pd.DataFrame: Loaded and formatted dataset.
    """
    try:
        data = pd.read_csv(Path(dataset_path) / file_name)
        # Format column names by removing spaces and converting to lowercase
        data.columns = data.columns.str.replace(' ', '_').str.lower()
        return data
    except FileNotFoundError as e:
        _logger.error(f"File {file_name} not found in {dataset_path}: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        _logger.error(f"File {file_name} is empty: {e}")
        raise
    except Exception as e:
        _logger.error(f"An error occurred while loading {file_name}: {e}")
        raise


def remove_old_pipeline(files_to_keep: t.List[str]) -> None:
    """
    Remove old pipelines to ensure a one-to-one mapping between the package version and the model version.

    Args:
        files_to_keep (List[str]): List of file names that should not be deleted.
    """
    do_not_delete = set(files_to_keep + ["__init__.py"])
    try:
        for file in trained_model_dir.iterdir():
            if file.name not in do_not_delete:
                file.unlink()
                _logger.info(f"Deleted old pipeline: {file.name}")
    except FileNotFoundError as e:
        _logger.error(f"File not found while attempting to remove old pipelines: {e}")
    except Exception as e:
        _logger.error(f"An error occurred while removing old pipelines: {e}")


def save_pipeline(
    pipeline_to_persist: Pipeline, 
    study_to_persist: optuna.study.Study, 
    model_to_persist: xgb.XGBRegressor
) -> None:
    """
    Save the versioned pipeline, Optuna study, and XGBoost model.

    Args:
        pipeline_to_persist (Pipeline): The pipeline to save.
        study_to_persist (optuna.study.Study): The study to save.
        model_to_persist (xgb.XGBRegressor): The model to save.
    """
    try:
        pipeline_name = f"{config.artifacts.pipeline_saved}_v{_version}.pkl"
        study_name = f"{config.artifacts.study_saved}_v{_version}.pkl"
        model_name = f"{config.artifacts.best_model_saved}_v{_version}.pkl"
        model_tar_name = f"{config.artifacts.best_model_saved}_v{_version}.tar.gz"

        remove_old_pipeline(files_to_keep=[pipeline_name, study_name, model_name, model_tar_name])

        pipeline_save_path = trained_model_dir / pipeline_name
        study_save_path = trained_model_dir / study_name
        model_save_path = trained_model_dir / model_name

        joblib.dump(pipeline_to_persist, pipeline_save_path)
        joblib.dump(study_to_persist, study_save_path)
        joblib.dump(model_to_persist, model_save_path)

        # Save model in tar.gz format
        with tarfile.open(trained_model_dir / model_tar_name, "w:gz") as tar:
            tar.add(model_save_path, arcname=model_name)

        _logger.info(f"Pipeline saved: {pipeline_name}")
        _logger.info(f"Study saved: {study_name}")
        _logger.info(f"Model saved: {model_name}")

    except Exception as e:
        _logger.error(f"An error occurred while saving the pipeline, study, or model: {e}")
        raise


def load_pipeline(pipeline_name: str, study_name: str, model_name: str) -> t.Tuple[Pipeline, optuna.study.Study, xgb.XGBRegressor]:
    """
    Load the versioned pipeline, Optuna study, and XGBoost model.

    Args:
        pipeline_name (str): The name of the saved pipeline file.
        study_name (str): The name of the saved study file.
        model_name (str): The name of the saved model file.

    Returns:
        Tuple[Pipeline, optuna.study.Study, xgb.XGBRegressor]: Loaded pipeline, study, and model.
    """
    try:
        pipeline_path = trained_model_dir / pipeline_name
        study_path = trained_model_dir / study_name
        model_path = trained_model_dir / model_name

        saved_pipeline = joblib.load(filename=pipeline_path)
        saved_study = joblib.load(filename=study_path)
        saved_model = joblib.load(filename=model_path)

        _logger.info(f"Pipeline loaded: {pipeline_name}")
        _logger.info(f"Study loaded: {study_name}")
        _logger.info(f"Model loaded: {model_name}")

        return saved_pipeline, saved_study, saved_model

    except FileNotFoundError as e:
        _logger.error(f"File not found while loading {pipeline_name}, {study_name}, or {model_name}: {e}")
        raise
    except Exception as e:
        _logger.error(f"An error occurred while loading the pipeline, study, or model: {e}")
        raise
