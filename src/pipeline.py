#!/usr/bin/env python3

import os
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sys

# Setting up the correct path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.core import config
import src.processing.preprocessors as pp

# Setting up logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

def create_pipeline() -> Pipeline:
    """
    Create a preprocessing pipeline for the model.
    
    Returns:
        Pipeline: The preprocessing pipeline.
    """
    _logger.info("Creating the data preprocessing pipeline...")

    # Categorical features imputation and encoding
    categorical_pipeline = Pipeline(steps=[
        ('cat_imputer', pp.CatImputer(features=config.variables.categorical_features)),  # Imputes missing values in categorical features
        ('ordinal_encoder', pp.OrdinalCatEncoder(features=config.variables.categorical_features))  # Ordinally encodes categorical features
    ])

    # Numerical features imputation and scaling
    numerical_pipeline = Pipeline(steps=[
        ('num_imputer', SimpleImputer(strategy='mean')),  # Imputes missing values in numerical features
        ('scaler', StandardScaler())  # Scales numerical features to have mean=0 and variance=1
    ])

    # Full preprocessing pipeline
    full_pipeline = Pipeline(steps=[
        ('cat_pipeline', categorical_pipeline),
        ('num_pipeline', numerical_pipeline)
    ])

    _logger.info("Pipeline creation complete.")
    return full_pipeline

# Create the preprocessing pipeline at the module level so that it can be easily imported
pipe_transformer = create_pipeline()

def main():
    # Paths for input and output
    input_path = "/opt/ml/processing/input/housing_price_dataset.csv"  
    output_path = "/opt/ml/processing/output/processed_data.csv"

    # Load input data
    _logger.info(f"Loading input data from {input_path}")
    df = pd.read_csv(input_path)

    # Apply the preprocessing pipeline to the data
    _logger.info("Applying the preprocessing pipeline to the data...")
    df_transformed = pipe_transformer.fit_transform(df)

    # Convert the transformed data to a DataFrame
    df_transformed = pd.DataFrame(df_transformed, columns=df.columns)

    # Save the processed data to the output path
    _logger.info(f"Saving processed data to {output_path}")
    # Ensure target column "price" is the first column
    target_column = "price"

    # Reorder columns if necessary
    columns = [target_column] + [col for col in df_transformed.columns if col != target_column]
    df_transformed = df_transformed[columns]
    df_transformed.to_csv(output_path, index=False)

    _logger.info("Preprocessing complete.")

if __name__ == "__main__":
    main()
