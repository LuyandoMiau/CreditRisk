"""
This file contains functions for data preparation and preprocessing.
This one should directly handle data from an already existing dataset, a CSV, PICKLE or EXCEL file.
"""

# Impport necessary libraries
import logging
from functools import wraps
from time import perf_counter
from typing import Callable, ParamSpec, TypeVar

import pandas as pd
import numpy as np
import sklearn
import sqlite3
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Literal

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def log_execution(func: Callable[P, R]) -> Callable[P, R]:
    """Log when an operation starts, finishes, and how long it takes."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        operation = func.__qualname__
        started_at = perf_counter()
        logger.info("Starting %s", operation)
        try:
            result = func(*args, **kwargs)
        except Exception:
            logger.exception("%s failed after %.3f seconds", operation, perf_counter() - started_at)
            raise
        logger.info("%s completed in %.3f seconds", operation, perf_counter() - started_at)
        return result

    return wrapper


"""
This class is responsible for reading data from a CSV, EXCEL or PKL file and storing it in a pandas DataFrame. It also provides a method to create a copy of the data to avoid modifying the original dataset.
"""
class DataReader:
    
    def __init__(self, file_path: str):
        
        """ 
        The init just requires the file path of the dataset to be read. It can be a CSV, EXCEL or PKL file. 
        """
        self.file_path = file_path
        self.data = None

    # READS the data from a CSV, EXCEL or PKL file and stores it in the data attribute.
    @log_execution
    def read_data(self):
        """
        Reads data from a CSV, Excel, or Pickle file and stores it in the data attribute.
        
        Supports file formats: .csv, .xlsx, .xls, .pkl
        Handles unclosed quotes (EOF inside string) and line tokenization errors cleanly.
        """
        path = Path(self.file_path)
        ext = path.suffix.lower()

        if ext == '.csv':
            logger.info("Reading CSV file: %s", self.file_path)
            try:
                # 1. Primary Attempt: Use Python engine with line skipping to bypass malformed quotes/rows
                self.data = pd.read_csv(
                    self.file_path,
                    engine='python',
                    on_bad_lines='skip',
                    encoding='utf-8-sig'
                )
            except (pd.errors.ParserError, Exception) as e:
                logger.warning("Primary CSV read failed (%s). Retrying without quote tokenization.", e)
                # 2. Fallback Attempt: Disable quote boundary parsing completely so unclosed quotes won't reach EOF
                self.data = pd.read_csv(
                    self.file_path,
                    engine='python',
                    quoting=csv.QUOTE_NONE,
                    on_bad_lines='skip',
                    encoding='utf-8-sig'
                )

            # Post-processing clean-up for trailing delimiter artifacts (e.g., inq_last_12m;;;;)
            self.data.columns = self.data.columns.str.rstrip(';')

        elif ext in ('.xlsx', '.xls'):
            logger.info("Reading Excel file: %s", self.file_path)
            self.data = pd.read_excel(self.file_path)

        elif ext == '.pkl':
            logger.info("Reading pickle file: %s", self.file_path)
            self.data = pd.read_pickle(self.file_path)

        else:
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats: .csv, .xlsx, .xls, .pkl")
        
        logger.info("Loaded dataset with %d rows and %d columns", *self.data.shape)
    
    # CREATES a copy of the data to avoid modifying the original dataset.   
    @log_execution
    def copy_data(self):
        
        """
        Returns a copy of the data to avoid modifying the original dataset.
        """
        return self.data.copy()
    
""" This class is responsible for exploring the data. 
It provides methods to explore the dataset and return some basic information about it.
It also provides a method to explore the dataset using sqlite extension and return some basic information about it.
"""    
class DataExplorerAdvanced:
    
    def __init__(self, data: pd.DataFrame):
        
        """
        The init requires the data to be explored. It should be a pandas DataFrame.
        """
        self.data = data
        
    # EXPLORE the dataset and return some basic information about it.
    @log_execution
    def explore_data(self):
        
        """
        Explores the dataset and returns some basic information about it.
        """
        logger.info("Data shape: %s", self.data.shape)
        logger.info("Data types:\n%s", self.data.dtypes.to_string())
        logger.info("Missing values by column:\n%s", self.data.isnull().sum().to_string())
        logger.info("Data description:\n%s", self.data.describe().to_string())
    
    # EXPLORE the dataset using sqlite extension and return some basic information about it.
    @log_execution
    def save_to_sqlite(self, db_path: str, table_name: str):
        """
        Saves the pandas DataFrame into a physical SQLite database file (.db) on disk.
        Creates the target directory automatically if it does not exist.
        """
        target_path = Path(db_path)
        
        # Ensure the destination directory exists before opening the SQLite connection
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect directly to the disk file path instead of in-memory RAM
        logger.info("Connecting to SQLite database file at: %s", target_path)
        conn = sqlite3.connect(target_path)

        try:
            # Write DataFrame to the database file
            logger.info("Writing %d rows to SQLite table '%s'", len(self.data), table_name)
            self.data.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info("Successfully saved database file to %s", target_path.resolve())
        finally:
            # Ensure connection is closed cleanly even if an exception occurs
            conn.close()

"""
This class is responsible for preprocessing the data. 
It provides methods to clear unnecessary columns and transform column types based on specified transformation types.
"""
class DataPreprocessor:
    
    def __init__(self, data: pd.DataFrame):
        
        """
        The init requires the data to be preprocessed. It should be a pandas DataFrame.
        """
        self.data = data
    
    # SAVE the column names of the dataset for later use.    
    @log_execution
    def column_names(self):
        
        """
        Returns the column names of the dataset.
        """
        return self.data.columns.tolist()
    
    # CLEAR the specified columns from the dataset =====> ["Unnamed: 0"]
    # this is not required for the preprocessing but it can be useful to remove unnecessary columns.
    @log_execution
    def clear_columns(self, columns_to_clear: List[str]):
        
        """
        Clears the specified columns from the dataset.
        
        Parameters:
        columns_to_clear (list): List of column names to be cleared.
        """
        logger.info("Dropping columns: %s", columns_to_clear)
        self.data.drop(columns=columns_to_clear, inplace=True)
        logger.info("Dataset now has %d columns", len(self.data.columns))

    # Handle some column types transformations for columns
    # But this is a particular case of our dataset, so thez are optional so variables/columns can be different for other cases
    @log_execution
    def transform_columns_types(self, transformation_type: str, columns: List[str], new_column_names: List[str]):
        
        """
        The function transforms a set of columns depending on the transformation type
        
        transformation types:
        - String_to_Numeric_specif: FOR OUR SPECIFIC DATASET, for columns with string values that can be converted to numeric values, but with specific cases to handle
        - String_to_Numeric_specif2: FOR OUR SPECIFIC DATASET, for columns with string values that can be converted to numeric values, but with specific cases to handle
        - String_to_Numeric: for columns with string values that can be converted to numeric values
        - String_to_Datetime: for columns with string values that can be converted to datetime values
        - Categorical_to_Numeric: for columns with categorical values that can be converted to numeric values
        """
        
        logger.info(
            "Applying transformation '%s' to columns: %s",
            transformation_type,
            columns,
        )
        # Here first case =====> Column: emp_length
        if transformation_type == "String_to_Numeric_specif":
            for column in columns:
                self.data[column] = (
                    self.data[column]
                        .replace({"< 1 year": "0", "n/a": "0"}) # Replace all of these 
                        .str.extract(r"(\d+)", expand=False)
                        .fillna(0) # fill nas with zeros
                        .astype(int)
                )
        
        # Here second case =====> Column: term: get rid off the word months
        elif transformation_type == "String_to_Numeric_specif2":
                    for column in columns:
                        self.data[column] = (
                            self.data[column]
                                .str.replace(" months", "", regex=False) # Replace " months" with ""
                                .fillna(0) # fill nas with zeros
                                .astype(int)
                        )
                        logger.info("Converted column '%s' from month strings to integers", column)

        # Here second case
        elif transformation_type == "String_to_Numeric":
            for i, column in enumerate(columns):
                self.data[new_column_names[i]] = pd.to_numeric(self.data[column], errors='coerce')
                logger.info("Converted '%s' to numeric column '%s'", column, new_column_names[i])

        # Here the thirs case =====> Column: earliest_cr_line
        elif transformation_type == "String_to_Datetime":
            for i, column in enumerate(columns):
                self.data[new_column_names[i]] = pd.to_datetime(self.data[column], format='%b-%y', errors='coerce')
                logger.info("Converted '%s' to datetime column '%s'", column, new_column_names[i])

        # Here the fourth case
        elif transformation_type == "Categorical_to_Numeric":
            for i, column in enumerate(columns):
                self.data[new_column_names[i]] = self.data[column].astype('category').cat.codes
                logger.info("Encoded '%s' as numeric column '%s'", column, new_column_names[i])
        
        # We get the defined reference date from the user input, and convert it to a datetime object
        @staticmethod
        def get_defined_reference_date(define_reference_date: str):
             defined_reference_date = pd.to_datetime(define_reference_date)
             return defined_reference_date
         
        # Specific function to get the days and months since the earliest credit line, based on the defined reference date
        @log_execution
        def get_days_months_since_earliest_credit_line(self, defined_reference_date: str):
            
            # Retrieve the defined reference date from the user input and convert it to a datetime object
            defined_reference_date = self.get_defined_reference_date(defined_reference_date)
            
            # Check that the column 'earliest_cr_line_date' exists in the DataFrame
            if 'earliest_cr_line_date' not in self.data.columns:
                raise ValueError("Column 'earliest_cr_line_date' does not exist in the DataFrame. Check in parameters in the part of data_preparation/data_preprocessor/transformations/new_column_names")
            
            # Get the days and months since the earliest credit line, based on the defined reference date
            self.data['days_since_earliest_credit_line'] = defined_reference_date - self.data['earliest_cr_line_date']
            self.data['months_since_earliest_credit_line'] = (self.data['days_since_earliest_credit_line'].dt.days / 30.44).round()
            
            # Also get some descriptive statistics for the new columns
            logger.info(
                "Days since earliest credit line:\n%s",
                self.data['days_since_earliest_credit_line'].describe().to_string(),
            )
            logger.info(
                "Months since earliest credit line:\n%s",
                self.data['months_since_earliest_credit_line'].describe().to_string(),
            )
            
        # Here a function to handle negative values for months_since_earliest_credit_line, in case the reference date is in the future, the negative values will be set to the maximum value of the column, and the user will get notified about it
        @log_execution
        def handle_negative_values(self):
            if 'months_since_earliest_credit_line' in self.data.columns:
                negative_values_count = (self.data['months_since_earliest_credit_line'] < 0).sum()
                min_value = self.data['months_since_earliest_credit_line'].min()
                logger.info(
                    "Minimum months since earliest credit line: %s",
                    min_value,
                )
                if negative_values_count > 0:
                    max_value = self.data['months_since_earliest_credit_line'].max()
                    self.data.loc[self.data['months_since_earliest_credit_line'] < 0, 'months_since_earliest_credit_line'] = max_value
                    logger.warning(
                        "Replaced %d negative months-since-earliest-credit-line values with the column maximum (%s)",
                        negative_values_count,
                        max_value,
                    )
                else:
                    logger.info("No negative months-since-earliest-credit-line values found")
            else:
                logger.info("Skipping negative-value handling; target column is absent")
        
        # Get days and months since issue_date, based on the defined reference date
        @log_execution
        def get_days_months_since_issue_date(self, defined_reference_date: str):
            # Retrieve the defined reference date from the user input and convert it to a datetime object
            defined_reference_date = self.get_defined_reference_date(defined_reference_date)
            
            # Check that the column 'issue_d_date' exists in the DataFrame
            if 'issue_d_date' not in self.data.columns:
                raise ValueError("Column 'issue_d_date' does not exist in the DataFrame. Check in parameters in the part of data_preparation/data_preprocessor/transformations/new_column_names")
            
            # Get the days and months since the issue date, based on the defined reference date
            self.data['days_since_issue_date'] = defined_reference_date - self.data['issue_d_date']
            self.data['months_since_issue_date'] = (self.data['days_since_issue_date'].dt.days / 30.44).round()
            
            # Also get some descriptive statistics for the new columns
            logger.info(
                "Days since issue date:\n%s",
                self.data['days_since_issue_date'].describe().to_string(),
            )
            logger.info(
                "Months since issue date:\n%s",
                self.data['months_since_issue_date'].describe().to_string(),
            )
        
        


        

        
    