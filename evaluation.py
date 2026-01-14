import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


def evaluate_score_cls(true, predicted):
    return {
        "accuracy": accuracy_score(true, predicted),
        "f1_score": f1_score(true, predicted, average='weighted')
    }


def evaluate_score_reg(true, predicted):
    mse = mean_squared_error(true, predicted)
    return {
        "MAE": mean_absolute_error(true, predicted),
        "RMSE": np.sqrt(mse),
        "r2_score": r2_score(true, predicted)
    }
