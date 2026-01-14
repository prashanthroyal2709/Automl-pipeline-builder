import pandas as pd
import os
def data_ingestion(file_path,target_col):
    """
    Load a dataset from CSV or Excel and split into features (X) and target (y).

    Parameters:
    - file_path (str): Path to the CSV or Excel file
    - target_col (str): Name of the target column

    Returns:
    - X (pd.DataFrame): Feature variables
    - y (pd.Series): Target variable
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext==".csv":
        df=pd.read_csv(file_path)
    elif ext in [".xls",".xlsx"]:
        df=pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format! Only CSV and Excel files are allowed.")

        
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset!")

    numerical_cols=df.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols=df.select_dtypes(include='O').columns.tolist()
    for cols in numerical_cols:
        if df[cols].isnull().sum()>0:
            df[cols]=df[cols].fillna(df[cols].median())

    for cols in categorical_cols:
        if df[cols].isnull().sum()>0:
            df[cols]=df[cols].fillna(df[cols].mode()[0])

    for cols in numerical_cols:
        q1=df[cols].quantile(0.25)
        q3=df[cols].quantile(0.75)
        iqr=q3-q1
        lower=q1-(1.5*iqr)
        upper=q3+(1.5*iqr)
        df[cols]=df[cols].clip(lower,upper)

    X=df.drop(target_col,axis=1)
    y=df.loc[:,target_col]
    return X,y