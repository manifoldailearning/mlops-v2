# This file contains utility functions for preprocessing and modeling.

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import shap
from matplotlib import pyplot as plt


def model_fit_xgb(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_importance=False,
    if_shap=True,
    multi_shap=False,
    verbose=True,
    sample_percent=1.0,
):
    """
    Fits an XGBoost model to the provided training data.

    Args:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series or list): Training target values.
        X_test (pd.DataFrame): Testing feature set.
        y_test (pd.Series or list): Testing target values.
        feature_importance (bool): Whether to plot feature importance.
        if_shap (bool): Whether to compute and plot SHAP values.
        multi_shap (bool): Whether to plot SHAP bar plot.
        verbose (bool): Whether to print model evaluation metrics.
        sample_percent (float): Percentage of data to use for SHAP calculations.

    Returns:
        xgb_model (xgb.XGBRegressor): The trained XGBoost model.
        metrics (pd.DataFrame): Model performance metrics.
    """
    
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise ValueError("X_train and X_test should be pandas DataFrames.")
    if not (0.0 <= sample_percent <= 1.0):
        raise ValueError("sample_percent must be between 0.0 and 1.0")

    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=5)
    xgb_model.fit(X_train, y_train)

    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)

    train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    metrics = pd.DataFrame(
        {
            "Train": [train_rmse, train_mae, train_r2],
            "Test": [test_rmse, test_mae, test_r2],
        },
        index=["RMSE", "MAE", "R2"],
    ).round(4)

    if verbose:
        print(metrics)

    if feature_importance:
        plt.rcParams["figure.figsize"] = [10, 5]
        xgb.plot_importance(xgb_model, importance_type="gain", max_num_features=30)
        plt.show()

    if if_shap:
        try:
            valid_sample = X_train.sample(int(np.ceil(len(X_train) * sample_percent)))
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer(valid_sample)
            shap.plots.beeswarm(shap_values, max_display=30, show=False)
            top_features = [t.get_text() for t in reversed(plt.gca().yaxis.get_majorticklabels())]
            plt.show()

            if multi_shap:
                shap.plots.bar(shap_values, max_display=30)

            return xgb_model, top_features, metrics

        except Exception as e:
            print(f"Error during SHAP analysis: {e}")

    return xgb_model, metrics


def cv_fit_xgb(
    df,
    cat_to_encode,
    other_cat,
    num_cols,
    transformers="empty",
    n_fold=5,
    target=None,
):
    """
    Performs n-fold cross-validation to fit an XGBoost model with feature engineering.

    Args:
        df (pd.DataFrame): The entire dataset.
        cat_to_encode (list): Categorical variables to encode.
        other_cat (list): Other categorical variables.
        num_cols (list): Numerical variables.
        transformers (str or sklearn.pipeline.Pipeline): Transformers to apply.
        n_fold (int): Number of cross-validation folds.
        target (str): Target variable.

    Returns:
        metrics_list (list of pd.DataFrame): Metrics from each fold.
    """

    df = df.copy()
    metrics_list = []
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=5)

    for t_index, v_index in kf.split(df):
        X_t = df.iloc[t_index][cat_to_encode + other_cat + num_cols]
        y_t = df.iloc[t_index][target]
        X_v = df.iloc[v_index][cat_to_encode + other_cat + num_cols]
        y_v = df.iloc[v_index][target]

        if transformers != "empty":
            transformers.fit(X_t, y_t)
            X_t = transformers.transform(X_t)
            X_v = transformers.transform(X_v)

        _, metrics = model_fit_xgb(
            X_train=X_t,
            X_test=X_v,
            y_train=y_t,
            y_test=y_v,
            feature_importance=False,
            if_shap=False,
            multi_shap=False,
            verbose=True,
        )
        metrics_list.append(metrics)

    return metrics_list


def plot_xy(df, var, target=None, bar_width=0.8, no_plot=False):
    """
    Plots the average of the target variable against a given feature.

    Args:
        df (pd.DataFrame): The dataset.
        var (str): Feature to plot against the target.
        target (str): Target variable.
        bar_width (float): Width of the bars in the plot.
        no_plot (bool): If True, no plot will be generated.

    Returns:
        pd.DataFrame: Aggregated data if no_plot is True.
    """

    if no_plot:
        return df[[var, target]].groupby(var).agg({var: 'count', target: 'mean'}).rename(columns={var: 'count', target: 'mean'}).sort_index()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    x = sorted(df[var].dropna().unique())
    y1 = df[[var, target]].groupby(var).agg({target: 'mean'}).rename(columns={target: 'mean'}).sort_index()
    y2 = df[[var, target]].groupby(var).agg({var: 'count'}).rename(columns={var: 'count'}).sort_index().values.flatten()

    ax1.plot(x, y1, 'g-')
    ax2.bar(x, y2, alpha=0.2, width=bar_width)

    ax1.set_xlabel(var)
    ax1.set_ylabel('AxT', color='g')
    ax2.set_ylabel('Count', color='b')

    plt.show()
