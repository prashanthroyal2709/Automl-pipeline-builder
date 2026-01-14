from data_ingestion import data_ingestion
from feature_engineering import FeatureEngineering
from preprocessing import create_preprocessing_pipeline
from model_trainer import model_train
from evaluation import evaluate_score_cls, evaluate_score_reg
from hyperparameter import *

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB


def run_automl(df, target_col):
    """
    Runs full AutoML pipeline and returns best model & score
    """

    # -----------------------------
    # Step 1: Split features & target
    # -----------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # -----------------------------
    # Step 2: Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Step 3: Preprocessing
    # -----------------------------
    preprocessing_pipeline = create_preprocessing_pipeline(X_train)

    # -----------------------------
    # Step 4: Detect problem type
    # -----------------------------
    is_classification = (y_train.dtype == "O") or (y_train.nunique() <= 10)

    # -----------------------------
    # Step 5: Models & hyperparams
    # -----------------------------
    if is_classification:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier(),
            "SVC": SVC(),
            "NaiveBayes": GaussianNB(),
            "KNeighbors": KNeighborsClassifier()
        }
        param_grids = {
            "LogisticRegression": log_reg_params,
            "DecisionTree": dt_params,
            "SVC": svc_params,
            "KNeighbors": knc_params,
            "RandomForest": random_params,
            "NaiveBayes": naive_params
        }
    else:
        models = {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(),
            "RandomForest": RandomForestRegressor(),
            "SVR": SVR(),
            "KNeighbors": KNeighborsRegressor()
        }
        param_grids = {
            "LinearRegression": linreg_params,
            "DecisionTree": dtr_params,
            "RandomForest": rfr_params,
            "SVR": svr_params,
            "KNeighbors": knr_params
        }

    # -----------------------------
    # Step 6: Train & evaluate
    # -----------------------------
    results = []

    for name, model in models.items():
        grid = param_grids.get(name)

        pipeline = model_train(
            preprocessing_pipeline,
            model,
            param_grid=grid,
            X_train=X_train,
            y_train=y_train,
            is_classification=is_classification
        )

        y_pred = pipeline.predict(X_test)

        if is_classification:
            metrics = evaluate_score_cls(y_test, y_pred)
            score = metrics["f1_score"]
        else:
            metrics = evaluate_score_reg(y_test, y_pred)
            score = metrics["r2_score"]

        results.append({
            "model_name": name,
            "score": score,
            "pipeline": pipeline
        })

    # -----------------------------
    # Step 7: Best model
    # -----------------------------
    results_df = pd.DataFrame(results)
    best_row = results_df.sort_values("score", ascending=False).iloc[0]

    best_pipeline = best_row["pipeline"]
    best_score = best_row["score"]

    return best_pipeline, best_score, results_df
