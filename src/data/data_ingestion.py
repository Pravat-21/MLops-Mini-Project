import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from src.logger import file_logging, console_logging
from src.utils import load_params,load_data,save_data
#from src.exception import CustomException
from src.exception import CustomException

file_logger=file_logging("Data Ingestion")



def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        file_logger.info('Data preprocessing completed')
        return final_df
    except Exception as e:
        file_logger.error(f"Error has been occured into preprocessing funcstion in data ingestion module & the error is {e}")
        raise CustomException(e,sys)
    
def main():
    """This is the main function for data ingestion"""

    file_logger.info("Now in Data Ingestion module....")
    try:
        url='https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        df=load_data(url)
        file_logger.info("successfully load data.")

        final_df=preprocess_data(df)
        file_logger.info("successfully preprocessed data.")

        params=load_params()
        test_size=params['data_ingestion']['test_size']

        train_data,test_data= train_test_split(final_df, test_size=test_size, random_state=42)

        save_data(train_data,file_path="./data/raw/train_raw.csv")
        save_data(test_data,file_path="./data/raw/test_raw.csv")
        file_logger.info("successfully creted raw data & add to raw folder.")

    except Exception as e:
        file_logger.error(f"Error has been occured into main function in data ingestion module & the error is {e}")
        raise CustomException(e,sys)


if __name__=="__main__":

    main()
    print("Everything of Data ingestion is done")