import numpy as np
import pandas as pd
import pickle
import mlflow
import json
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.logger import file_logging, console_logging
from src.utils import load_params,load_data,save_data
from src.exception import CustomException
import dagshub
import os
#--------------------------------------Configuration------------------------------------------
file_logger=file_logging("Model Evaluation")
#mlflow.set_tracking_uri('https://dagshub.com/Pravat-21/MLops-Mini-Project.mlflow')
#dagshub.init(repo_owner='Pravat-21', repo_name='MLops-Mini-Project', mlflow=True)

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Pravat-21"
repo_name = "MLops-Mini-Project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

#-------------------------------------Functions-----------------------------------------------
def load_model(file_path: str):
    """Load the trained model from a file."""
    file_logger.info("In load_model function from model_evaluation module....")
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)

        file_logger.info(f'Model loaded from {file_path}')
        return model
    except Exception as e:
        file_logger.error(f"In load_model function from model Evaluation, error has been ocurred & error is {e}")
        raise CustomException(e,sys)
    
def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    file_logger.info("In evaluate_model function from model_evaluation module....")

    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

        file_logger.info('Model evaluation metrics calculated')
        return metrics_dict
    
    except Exception as e:
        file_logger.error(f"In evaluate_model function from model Evaluation, error has been ocurred & error is {e}")
        raise CustomException(e,sys)
    
def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    file_logger.info("In save_metrics function from model_evaluation module....")

    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)

        file_logger.info(f'Metrics saved to {file_path}')

    except Exception as e:
        file_logger.error(f"In save_metrics function from model Evaluation, error has been ocurred & error is {e}")
        raise CustomException(e,sys)
    
def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        file_logger.info('Model info saved to %s', file_path)
    except Exception as e:
        file_logger.error('Error occurred while saving the model info: %s', e)
        raise
    
#----------------------------------------Main function-----------------------------------------------
def main():
    """This is the main function for Model Evaluation"""
    file_logger.info("Now in Model Evaluation module....")

    mlflow.set_experiment("dvc-pipeline")
    with mlflow.start_run() as run:  # Start an MLflow run

        try:
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)

            save_metrics(metrics, 'reports/metrics.json')
            file_logger.info("Model_evaluation module has been successfully completed")

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(clf, "model")
            
            # Save model info
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            file_logger.error(f"In model Evaluation module, error has been ocurred & error is {e}")
            raise CustomException(e,sys)
    
#--------------------------------------------------------------------------------------------------
if __name__=="__main__":

    main()
    print("Everything of Model Evaluation is done")











