from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from feature_engineering import FeatureEngineering

def model_train(preprocessing_pipeline, model, param_grid=None, 
                X_train=None, y_train=None, 
                is_classification=True, n_iter=20, cv=5, random_state=42):
    """
    Train a model using a pipeline with optional RandomizedSearchCV for hyperparameter tuning.

    Parameters:
    - preprocessing_pipeline: ColumnTransformer
    - model: sklearn model object
    - param_grid: dict of hyperparameters for RandomizedSearchCV
    - X_train, y_train: training data
    - is_classification: True if classification problem, False for regression
    - n_iter: number of iterations for RandomizedSearchCV
    - cv: number of cross-validation folds
    - random_state: random seed

    Returns:
    - pipeline: trained pipeline (best model if hyperparameter tuning is used)
    """

    # Create the pipeline
    pipeline = Pipeline([
        ("feature_engineering", FeatureEngineering()),
        ("preprocessing_pipeline", preprocessing_pipeline),
        ("model", model)
    ])

    # Apply RandomizedSearchCV if param_grid is provided
    if param_grid:
        scoring = 'f1_weighted' if is_classification else 'r2'
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=random_state,
            verbose=1
        )
        search.fit(X_train, y_train)
        return search.best_estimator_  # return the best pipeline
    else:
        pipeline.fit(X_train, y_train)
        return pipeline
