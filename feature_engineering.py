from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Custom transformer to perform feature engineering:
    - Compute Rooms_per_sqfeet
    - Encode House_age ("New":0, "middle_aged":1, "Old":2)
    """

    def fit(self, X, y=None):
        # Nothing to fit; just return self
        return self

    def transform(self, X):
        X = X.copy()

        # Create Rooms_per_sqfeet if required columns exist
        if all(col in X.columns for col in ["Square_Footage", "Num_Bedrooms", "Num_Bathrooms"]):
            X["Rooms_per_sqfeet"] = (X["Num_Bedrooms"] + X["Num_Bathrooms"]) / X["Square_Footage"]

        # Encode House_age if column exists
        if "House_age" in X.columns:
            X["House_age"] = X["House_age"].map({
                "New": 0,
                "middle_aged": 1,
                "Old": 2
            })

        return X
