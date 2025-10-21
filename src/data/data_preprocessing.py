import numpy as np
import pandas as pd
import os
import sys
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import file_logging, console_logging
from src.utils import load_params,load_data,save_data
from src.exception import CustomException

#------------------------configuration----------------------------------------------------
file_logger=file_logging("Data Preprocessing")

nltk.download('wordnet')
nltk.download('stopwords')

#----------------------------- Functions--------------------------------------------------

def lemmatization(text):
    """Lemmatize the text."""
    file_logger.info("In lemmatization function from data preprocessing module....")

    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(word) for word in text]

        file_logger.info("lemmatization has been successfully done.....")
        return " ".join(text)
    
    except Exception as e:
        file_logger.error(f"In lemmatization function from data preprocessing error has been ocurred & error is {e}")
        raise CustomException(e,sys)
        

def remove_stop_words(text):
    """Remove stop words from the text."""
    file_logger.info("In remove_stop_words function from data preprocessing module....")

    try:
        stop_words = set(stopwords.words("english"))
        text = [word for word in str(text).split() if word not in stop_words]

        file_logger.info("remove_stop_words has been successfully done.....")
        return " ".join(text)
    
    except Exception as e:
        file_logger.error(f"In remove_stop_words function from data preprocessing error has been ocurred & error is {e}")
        raise CustomException(e,sys)

def removing_numbers(text):
    """Remove numbers from the text."""
    file_logger.info("In removing_numbers function from data preprocessing module....")

    try:
        text = ''.join([char for char in text if not char.isdigit()])

        file_logger.info("removing_numbers has been successfully done.....")
        return text
    
    except Exception as e:
        file_logger.error(f"In removing_numbers function from data preprocessing error has been ocurred & error is {e}")
        raise CustomException(e,sys)

def lower_case(text):
    """Convert text to lower case."""
    file_logger.info("In lower_case function from data preprocessing module....")

    try:
        text = text.split()
        text = [word.lower() for word in text]

        file_logger.info("lower_case has been successfully done.....")
        return " ".join(text)
    
    except Exception as e:
        file_logger.error(f"In lower_case function from data preprocessing error has been ocurred & error is {e}")
        raise CustomException(e,sys)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    file_logger.info("In removing_punctuations function from data preprocessing module....")

    try:
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text).strip()

        file_logger.info("removing_punctuations has been successfully done.....")
        return text
    
    except Exception as e:
        file_logger.error(f"In removing_punctuations function from data preprocessing error has been ocurred & error is {e}")
        raise CustomException(e,sys)

def removing_urls(text):
    """Remove URLs from the text."""
    file_logger.info("In removing_urls function from data preprocessing module....")

    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')

        file_logger.info("removing_urls has been successfully done.....")
        return url_pattern.sub(r'', text)
    
    except Exception as e:
        file_logger.error(f"In removing_urls function from data preprocessing error has been ocurred & error is {e}")
        raise CustomException(e,sys)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    file_logger.info("In remove_small_sentences function from data preprocessing module....")
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan

        file_logger.info("remove_small_sentences has been successfully done.....")
    
    except Exception as e:
        file_logger.error(f"In remove_small_sentences function from data preprocessing error has been ocurred & error is {e}")
        raise CustomException(e,sys)

def normalize_text(df):
    """Normalize the text data."""
    file_logger.info("In normalize_text function from data preprocessing module....")

    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        
        df['content'] = df['content'].apply(removing_numbers)
        
        df['content'] = df['content'].apply(removing_punctuations)
        
        df['content'] = df['content'].apply(removing_urls)
        
        df['content'] = df['content'].apply(lemmatization)
        
        file_logger.info("normalize_text has been successfully done.....")   
        return df
    
    except Exception as e:
        file_logger.error(f"In normalize_text function from data preprocessing error has been ocurred & error is {e}")
        raise CustomException(e,sys)

#--------------------------------------Main function----------------------------------------

def main():
    """This is the main function for data preprocessing"""
    file_logger.info("Now in Data preprocessing module....")

    try:
        # Fetch the data from data/raw
        file_logger.info("Fetching all the data from data/raw folder....")

        train_data=load_data('./data/raw/train_raw.csv')
        test_data=load_data('./data/raw/test_raw.csv')

        file_logger.info("Successfully fetched data from data/raw folder......")

        #filling missing values
        file_logger.info("filling missing values in train & test data....")

        #train_data.fillna('',inplace=True)
        #test_data.fillna('',inplace=True)

        #train_data = train_data.fillna('').astype(str)
        #test_data = test_data.fillna('').astype(str)

        file_logger.info("Successfully filled missing value with ''......")

        # Transform the data
        file_logger.info("Normalizing train & test data....")

        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        file_logger.info("Successfully normalized train & test data......")

        # Store the data inside data/interim
        file_logger.info("Storing train & test data into interim folder....")

        

        save_data(train_processed_data,file_path="./data/interim/train_processed.csv")
        save_data(test_processed_data,file_path="./data/interim/test_processed.csv")

        file_logger.info("successfully stored preprocessed data into data/exterim folder")
    
    except Exception as e:
        file_logger.error(f"Error has been occured into main function in data preprocessing module & the error is {e}")
        raise CustomException(e,sys)

#-------------------------------------------------------------------------------------------

if __name__=="__main__":

    main()
    print("Everything of Data Preprocessing is done")
