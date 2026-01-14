# AutoML Pipeline Builder

A generic AutoML pipeline that automates model training, evaluation, and deployment for regression and classification tasks.

## Features
- Supports CSV and Excel files
- Automatic detection of regression/classification
- Preprocessing: scaling, encoding, missing values
- Trains multiple models with hyperparameter tuning
- Selects best model based on cross-validation
- Saves model and metrics
- Streamlit app for interactive predictions
- Modular and reusable code

## Project Structure
app.py
main.py
data_ingestion.py
feature_engineering.py
preprocessing.py
model_trainer.py
evaluation.py
hyperparameter.py
requirements.txt
artifacts/ # optional: saved models and metrics


## Run Locally
```bash
git clone https://github.com/<your-username>/Automl-pipeline-builder.git
cd Automl-pipeline-builder
pip install -r requirements.txt
python main.py
streamlit run app.py


Dependencies
pandas
numpy
scikit-learn
joblib
streamlit
matplotlib
seaborn
openpyxl
