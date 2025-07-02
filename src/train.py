import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import argparse
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def eval_metrics(actual, pred, pred_proba):
    """Calculate and return a dictionary of evaluation metrics."""
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    roc_auc = roc_auc_score(actual, pred_proba)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

def main(model_name):
    """
    Main function to run the model training and tracking workflow.
    """
    print(f"--- Starting Training for Model: {model_name} ---")

    # Load processed data
    processed_data_path = os.path.join('data', 'processed', 'model_input_data.csv')
    try:
        data = pd.read_csv(processed_data_path, index_col='CustomerId')
        print("Model input data loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Processed data not found at {processed_data_path}. Please run data_processing.py first.")
        return

    # Split data into training and testing sets
    X = data.drop('is_high_risk', axis=1)
    y = data['is_high_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Model and Hyperparameter Grid Definition ---
    if model_name.lower() == 'logistic_regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga']
        }
    elif model_name.lower() == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    else:
        print(f"ERROR: Model '{model_name}' not recognized. Choose 'logistic_regression' or 'random_forest'.")
        return

    # Set up MLflow experiment
    mlflow.set_experiment("Credit Scoring Model Comparison")

    with mlflow.start_run(run_name=f"train_{model_name}"):
        print("MLflow run started.")
        
        # --- Hyperparameter Tuning with GridSearchCV ---
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")
        
        # Log best parameters to MLflow
        mlflow.log_params(grid_search.best_params_)

        # Evaluate the best model on the test set
        predictions = best_model.predict(X_test)
        pred_probas = best_model.predict_proba(X_test)[:, 1] 

        # Calculate metrics
        metrics = eval_metrics(y_test, predictions, pred_probas)
        print("Evaluation Metrics on Test Set:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_name", model_name)

        # Log the model artifact to MLflow
        tracking_uri = mlflow.get_tracking_uri()
        print(f"Current tracking URI: {tracking_uri}")

        mlflow.sklearn.log_model(best_model, "model")
        print("Model logged to MLflow.")

        # --- Register the Best Model ---
        if model_name == 'random_forest' and metrics['f1_score'] > 0.5: # Example threshold
             # The `registered_model_name` is how we will find it later.
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="credit_risk_model",
                registered_model_name="CreditRiskModel-RandomForest"
            )
            print("Model registered in MLflow Model Registry.")

    print(f"--- Training for {model_name} Complete ---")


if __name__ == '__main__':
    # Use argparse to allow running from the command line
    parser = argparse.ArgumentParser(description="Train a credit risk model.")
    parser.add_argument(
        "model_name", 
        type=str, 
        choices=['logistic_regression', 'random_forest'], 
        help="The name of the model to train."
    )
    args = parser.parse_args()
    main(args.model_name)