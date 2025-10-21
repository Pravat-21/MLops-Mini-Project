import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import yaml
import sys
from src.logger import file_logging, console_logging
from src.utils import load_params,load_data,save_data
from src.exception import CustomException

#------------------------------------Configuration--------------------------------------------------
file_logger=file_logging("Model Building")

#-------------------------------------Functions-----------------------------------------------------

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> LogisticRegression:
    """Train the Gradient Boosting model."""
    file_logger.info("In train_model function from model_building module....")

    try:
        #clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l2')

        clf.fit(X_train, y_train)

        file_logger.info('Model training completed')
        return clf
    
    except Exception as e:
        file_logger.error(f"In train_model function from model Building, error has been ocurred & error is {e}")
        raise CustomException(e,sys)

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        
        file_logger.info(f'Model saved to {file_path}')
    
    except Exception as e:
        file_logger.error(f"In save_model function from model Building, error has been ocurred & error is {e}")
        raise CustomException(e,sys)
    
#----------------------------------------Main function-------------------------------------------------

def main():
    """This is the main function for Model Building"""
    file_logger.info("Now in Model Building module....")

    try:
        #parameters loading
        params=load_params()['model_building']

        train_data = load_data('./data/processed/train_bow.csv')

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)
        save_model(clf, 'models/model.pkl')

        file_logger.info("Model training has been done successfully in model building module")

    except Exception as e:
        file_logger.error(f"In model building module, error has been ocurred & error is {e}")
        raise CustomException(e,sys)

#--------------------------------------------------------------------------------------------
if __name__=="__main__":

    main()
    print("Everything of Model building is done")
