# This file contains the code for the preprocessing of the data.

import os
import sys
import pandas as pd
import numpy as np
import typing as t
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # Import to enable IterativeImputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV

class CatImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values in categorical columns with a default value ('missing').

    Args:
        features (list of str): List of features to be imputed.

    Attributes:
        features (list of str): The features to be imputed.
    """

    def __init__(self, features=None) -> None:
        if features is None:
            self.features = []
        elif not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CatImputer':
        """
        Fit the imputer on the dataset. (No fitting necessary for constant imputation)

        Args:
            X (pd.DataFrame): Input data to fit the imputer.
            y (pd.Series, optional): Ignored, exists for compatibility.

        Returns:
            self: Fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset by filling missing values.

        Args:
            X (pd.DataFrame): Input data to transform.

        Returns:
            pd.DataFrame: Transformed data with missing values filled.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        missing_features = [feat for feat in self.features if feat not in X.columns]
        if missing_features:
            raise ValueError(f"The following features are missing from the DataFrame: {missing_features}")

        X = X.copy()
        for feat in self.features:
            X[feat] = X[feat].fillna('missing')

        return X


class OrdinalCatEncoder(BaseEstimator, TransformerMixin):
    """
    Ordinal encoder for encoding categorical variables into numerical values.

    Args:
        features (list of str): List of features to be encoded.

    Attributes:
        features (list of str): The features to be encoded.
        ordinal (OrdinalEncoder): The fitted OrdinalEncoder instance.
    """

    def __init__(self, features=None) -> None:
        if features is None:
            self.features = []
        elif not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'OrdinalCatEncoder':
        """
        Fit the ordinal encoder to the dataset.

        Args:
            X (pd.DataFrame): Input data to fit the encoder.
            y (pd.Series, optional): Ignored, exists for compatibility.

        Returns:
            self: Fitted encoder.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        missing_features = [feat for feat in self.features if feat not in X.columns]
        if missing_features:
            raise ValueError(f"The following features are missing from the DataFrame: {missing_features}")

        self.ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.ordinal.fit(X[self.features])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset by encoding the specified features.

        Args:
            X (pd.DataFrame): Input data to transform.

        Returns:
            pd.DataFrame: Transformed data with encoded features.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        missing_features = [feat for feat in self.features if feat not in X.columns]
        if missing_features:
            raise ValueError(f"The following features are missing from the DataFrame: {missing_features}")

        X = X.copy()
        X_transformed = pd.DataFrame(
            data=self.ordinal.transform(X[self.features]),
            index=X.index,
            columns=self.features
        )

        # Replacing the original categorical columns with their encoded values
        X[self.features] = X_transformed

        return X


