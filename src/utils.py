import yaml
import os
import sys
from src.exception import CustomException
import pandas as pd
from src.logger import file_logging, console_logging

file_logger=file_logging("Utils logging")

def load_params()->dict:
    """Using this function we can load our params.yaml file for parameter usages"""

    file_logger.info("Now in load_params function from utils.py")
    try:
        with open("params.yaml","rb") as file:
            params=yaml.safe_load(file)
            file_logger.info("successfully load all the parameters.")
            return params
    except Exception as e:
        file_logger.error("Error has been occured in load_params function from utils.py")
        raise CustomException(e,sys)
    



    
def load_data(data_path:str)->pd.DataFrame:
    """Using this function we can load data located into given data path."""

    file_logger.info("Now in load_data function from utils.py")
    try:
        df=pd.read_csv(data_path)
        file_logger.info(f"successfully load the data from the {data_path}")
        return df
    except Exception as e :
        file_logger.error("Error has been occured in load_data function from utils.py")
        raise CustomException(e,sys)
    



    
def save_data(df:pd.DataFrame,file_path:str)->None:
    """This function helps to save csv data in the given folder with given file name"""

    file_logger.info("Now in save_data function from utils.py")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        df.to_csv(file_path,index=False)
        file_logger.info(f"successfully save the data into {file_path} as csv")

    except Exception as e:
        file_logger.error("Error has been occured in save_data function from utils.py")
        raise CustomException(e,sys)
    
    



    

    

