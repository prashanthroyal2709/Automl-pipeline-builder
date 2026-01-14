from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
def create_preprocessing_pipeline(X):
    """
    Create a ColumnTransformer to scale numerical columns and one-hot encode categorical columns.

    Parameters:
    - X (pd.DataFrame): The feature DataFrame

    Returns:
    - preprocessing_pipeline (ColumnTransformer): ColumnTransformer ready to use in a pipeline
    """
    numerical_cols=X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols=X.select_dtypes(include='O').columns.tolist()
    oh_transformer=OneHotEncoder(handle_unknown='ignore')
    sc_transformer=StandardScaler()
    preprocessing_pipeline=ColumnTransformer(transformers=[
        ("num",sc_transformer,numerical_cols),
        ("cat",oh_transformer,categorical_cols)
    ])
    return preprocessing_pipeline