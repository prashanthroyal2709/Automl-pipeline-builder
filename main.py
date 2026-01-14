from data_ingestion import data_ingestion
from feature_engineering import FeatureEngineering
from preprocessing import create_preprocessing_pipeline
from model_trainer import model_train
from evaluation import evaluate_score_cls, evaluate_score_reg
from hyperparameter import *

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

# -----------------------------
# Step 1: Load dataset
# -----------------------------
file_path = input("Enter dataset path: ")
target_col = input("Enter target column: ")

X, y = data_ingestion(file_path, target_col)

# -----------------------------
# Step 2: Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Step 3: Preprocessing pipeline
# -----------------------------
preprocessing_pipeline = create_preprocessing_pipeline(X_train)

# -----------------------------
# Step 4: Determine problem type
# -----------------------------
is_classification = (y_train.dtype == 'O') or (y_train.nunique() <= 10)

# -----------------------------
# Step 5: Define models and param grids
# -----------------------------
if is_classification:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "SVC": SVC(),
        "NaiveBayes": GaussianNB(),
        "KNeighborsClassifier": KNeighborsClassifier()
    }
    param_grids = {
        "LogisticRegression": log_reg_params,
        "DecisionTree": dt_params,
        "SVC": svc_params,
        "KNeighborsClassifier": knc_params,
        "RandomForest": random_params,
        "NaiveBayes": naive_params
    }
else:
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(),
        "SVR": SVR(),
        "KNeighborsRegressor": KNeighborsRegressor()
    }
    param_grids = {
        "LinearRegression": linreg_params,
        "DecisionTree": dtr_params,
        "RandomForest": rfr_params,
        "SVR": svr_params,
        "KNeighborsRegressor": knr_params
    }

# -----------------------------
# Step 6: Train models
# -----------------------------
results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    grid = param_grids.get(name, None)
    pipeline = model_train(
        preprocessing_pipeline,
        model,
        param_grid=grid,
        X_train=X_train,
        y_train=y_train,
        is_classification=is_classification
    )
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    if is_classification:
        metrics = evaluate_score_cls(y_test, y_pred)
        score = metrics["f1_score"]
    else:
        metrics = evaluate_score_reg(y_test, y_pred)
        score = metrics["r2_score"]
    
    print(f"{name} metrics: {metrics}")
    
    results.append({
        "model_name": name,
        "score": score,
        "pipeline": pipeline
    })

# -----------------------------
# Step 7: Select best model
# -----------------------------
best_model = pd.DataFrame(results).sort_values("score", ascending=False).iloc[0]
print(f"\nBest model:\n{best_model}")

# -----------------------------
# Step 8: Save best model
# -----------------------------
os.makedirs("artifacts", exist_ok=True)
joblib.dump(best_model["pipeline"], "artifacts/best_model.pkl")
print("Best model saved to artifacts/best_model.pkl")

# -----------------------------
# Step 9: Save metrics
# -----------------------------
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("artifacts/model_metrics.csv", index=False)
print("Metrics saved to artifacts/model_metrics.csv")
