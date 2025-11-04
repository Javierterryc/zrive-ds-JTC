import os
import pandas as pd
from typing import Tuple, List
import joblib
import logging
import time


from functions import (_push_relevant_dataframe, 
                       _format_date_columns, 
                       _temporal_data_split, 
                       _feature_label_split)

logger = logging.getLogger(__name__)

class CustomModel:
    '''
    This is my custom class created for Zrive DS module_4 assesment.
    This class provides the following methods:
        1. load_data: loading data from a parquet or CSV file.
        2. preprocess_data: preprocessing data using custom functions (from utils.py).
        3. temporal_split: performing a temporal split into train/validation/test sets.
        4. feature_split: separating features (X) and target (y).
        5. train: raining the specified model with chosen parameters (as specified in the input 'event').
        6. save_model: saving the trained model to disk.

    Attributes
    ----------
    model_class : estimator class
        The machine learning model class to be instantiated (e.g., XGBClassifier).
    data : pd.DataFrame
        The raw dataset loaded from disk.
    filtered_data : pd.DataFrame
        The dataset after preprocessing.
    model : object
        The trained model instance.
    model_name : str
        Model name for the saved model file
        The default model name will be: "modelclass_push_yyyy_mm_dd" where yyyy_mm_dd is the training date.
    '''
    def __init__(self, *, model_class):
        self.model_class = model_class

    def load_data(self, input_data_path : str, *, csv : bool = False) -> pd.DataFrame:
        '''
        Loads a dataset from input_data_path and returns it as a df.
        If csv=False, file type will be "parquet"
        '''
        if not csv:
            self.data = pd.read_parquet(input_data_path, engine="fastparquet")
        else:
            self.data = pd.read_csv(input_data_path)
        return self.data
    
    def preprocess_data(self) -> pd.DataFrame:
        '''
        Apply preprocessing steps to raw data df from .load_data() with functions from utils.py
        _push_relevant_dataframe: filters df to retain only orders with 5 or more outcome=1
        _format_date_columns: convert date columns to datetime

        Returns the preprocessed (filtered) dataset
        '''
        if not hasattr(self, "data"):
            raise AttributeError("No attribute 'data' found, run .load_data() first.")
        self.filtered_data = (
            self.data
            .pipe(_push_relevant_dataframe)
            .pipe(_format_date_columns)
        )
        return self.filtered_data
    
    def temporal_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        Performs a temporal split in which:
        - train: first 70% of the data
        - val: next 20% of the data
        - test: last 10% of the data (corresponds to latest observations from the dataset)

        Returns train, val and test sets (pd.DataFrame)
        '''
        if not hasattr(self, "filtered_data"):
            raise AttributeError("No filtered_df found, run .preprocess_data() first.")

        train_df, val_df, test_df = _temporal_data_split(self.filtered_data)
        return train_df, val_df, test_df

    def feature_split(self, df, *, label_col : str):
        ''' 
        Performs split to the dataset in order to separate features (X) from target (y)
        '''
        X, y = _feature_label_split(df, label_col)
        return X, y
    
    def train(self, X, y, *, params, train_features: List[str]): 
        '''
        Fits the specified model along with its parameters
        '''
        self.model = self.model_class(**params)
        self.model.fit(X[train_features], y)

        timestamp = time.strftime("%Y_%m_%d-%H%M")
        model_type_name = self.model.__class__.__name__
        self.model_name = f"{model_type_name}_push_{timestamp}.pkl" 

        logger.info(f"Model: {self.model_name} trained successfully")

    def save_model(self,*,output_path : str):
        '''
        Saves the model object to an 'output_path' specified by the user
        '''
        if not hasattr(self, "model") or self.model is None:
            raise AttributeError("No model trained yet, run '.train(...)' first.")

        os.makedirs(output_path, exist_ok=True)
        saved_model_path = os.path.join(output_path, self.model_name)
        joblib.dump(self.model, saved_model_path) 
   
        logger.info(f"Model: {self.model_name} saved successfully to: '{saved_model_path}'")
        return saved_model_path