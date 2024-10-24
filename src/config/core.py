# Configuration file to load the configuration file

from pathlib import Path
import sys
import typing as t
import logging

from pydantic import BaseModel, ValidationError
from yaml import safe_load

# Setting up logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# Project root directory
package_root = Path(__file__).resolve().parents[2]
config_path = package_root / "src" / "config" / "config.yml"
dataset_path = package_root 
trained_model_dir = package_root / "src" / "trained_models"



class PathsConfig(BaseModel):
    data_dir: str
    artifacts_dir: str

    @property
    def train_validation_data(self) -> str:
        return str(Path(self.data_dir) / "housing_price_dataset.csv")
    
    @property
    def test_data(self) -> str:
        return str(Path(self.data_dir) / "housing_price_dataset_modified.csv")

class ArtifactsConfig(BaseModel):
    pipeline_saved: str
    study_saved: str
    best_model_saved: str

class VariablesConfig(BaseModel):
    target: str
    features: t.List[str]
    categorical_features: t.List[str]
    numerical_features: t.List[str]


class HyperparametersConfig(BaseModel):
    max_depth: int
    learning_rate: float
    n_estimators: int

class EnvironmentConfig(BaseModel):
    dev: t.Dict[str, t.Any]
    test: t.Dict[str, t.Any]

class TrainingParamsConfig(BaseModel):
    test_size: float
    random_state: int

class Config(BaseModel):
    package_name: str
    paths: PathsConfig
    artifacts: ArtifactsConfig
    variables: VariablesConfig
    training_params: TrainingParamsConfig  # Updated field name
    hyperparameters: HyperparametersConfig
    environment: EnvironmentConfig

    class Config:
        protected_namespaces = ()




def load_config(config_path: Path = config_path) -> Config:
    """
    Load the configuration file.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        Config: Loaded configuration as a Config object.
    """
    try:
        with open(config_path, "r") as file:
            config_dict = safe_load(file.read())
            _logger.info(f"Loaded config data: {config_dict}")
            return Config(**config_dict)
    except FileNotFoundError as e:
        _logger.error(f"Configuration file not found: {e}")
        sys.exit(1)
    except ValidationError as e:
        _logger.error(f"Validation error while loading configuration: {e}")
        _logger.error(f"Loaded configuration data: {config_dict}")  # Log the problematic config
        sys.exit(1)
    except Exception as e:
        _logger.error(f"Unexpected error while loading configuration: {e}")
        sys.exit(1)



config = load_config()

# Uncomment for debugging if needed
# _logger.info(config)
