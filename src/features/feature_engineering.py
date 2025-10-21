import numpy as np
import pandas as pd
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from src.logger import file_logging, console_logging
from src.utils import load_params,load_data,save_data
from src.exception import CustomException
import yaml

#------------------------------Configuration--------------------------------------------------------
file_logger=file_logging("Feature Engineering")


#------------------------------Functions------------------------------------------------------------

def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int)->tuple:
    """apply BoW (Bag of words) into datasets."""
    file_logger.info("In apply_bow function from feature engineering module....")

    try:
        vectorizer=CountVectorizer(max_features=max_features)

        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)

        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        X_train_bow=vectorizer.fit_transform(X_train)
        X_test_bow=vectorizer.transform(X_test)

        train_df=pd.DataFrame(X_train_bow.toarray())
        train_df['label']=y_train

        test_df=pd.DataFrame(X_test_bow.toarray())
        test_df['label']=y_test

        file_logger.info('Successfully Bag of Words applied and data transformed')
        return train_df, test_df
    
    except Exception as e:
        file_logger.error(f"In apply_bow function from feature engineering, error has been ocurred & error is {e}")
        raise CustomException(e,sys)
    
#----------------------------------Main Function------------------------------------------------

def main():
    """This is the main function for feature engineering"""
    file_logger.info("Now in feature engineering module....")

    try:
        params=load_params()
        max_features=params['feature_engineering']['max_features']
        file_logger.info("params.yaml file has been loaded successfully & max_features has been extracted.")

        train_data=load_data('./data/interim/train_processed.csv') 
        test_data=load_data('./data/interim/test_processed.csv')
        file_logger.info("Data (train & test) has been loaded into from data/interim folder...")

        train_df, test_df = apply_bow(train_data, test_data, max_features)

        save_data(train_df,file_path="./data/processed/train_bow.csv")
        save_data(test_df,file_path="./data/processed/test_bow.csv")
        file_logger.info("Data (train & test) has been saved into from data/processed folder...")
    
    except Exception as e:
        file_logger.error(f"Error has been occured into main function in feature enginnering module & the error is {e}")
        raise CustomException(e,sys)
    
#------------------------------------------------------------------------------------------------------
if __name__=="__main__":

    main()
    print("Everything of Feature engineering is done")


















