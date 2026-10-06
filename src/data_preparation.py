"""
This file contains functions for data preparation and preprocessing.
This one should directly handle data from an already existing dataset, a CSV, PICKLE or EXCEL file.
"""

# Impport necessary libraries
import pandas as pd
import numpy as np
import sklearn
import sqlite3
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Literal

"""
This class is responsible for reading data from a CSV, EXCEL or PKL file and storing it in a pandas DataFrame. It also provides a method to create a copy of the data to avoid modifying the original dataset.
"""
class DataReader:
    
    def __init__(self, file_path: str):
        
        """ 
        The init just requires the file path of the dataset to be read. It can be a CSV or EXCEL file. 
        """
        self.file_path = file_path
        self.data = None

    # READS the data from a CSV or EXCEL file and stores it in the data attribute.
    def read_data(self):
        
        """
        Reads data from a CSV or EXCEL file and stores it in the data attribute.
        """

        # Handle clases for the different file formats
        if self.file_path.endswith('.csv'):
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.endswith('.xlsx') or self.file_path.endswith('.xls'):
            self.data = pd.read_excel(self.file_path)
        elif self.file_path.endswith('.pkl'):
            self.data = pd.read_pickle(self.file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or EXCEL file.")
    
    # CREATES a copy of the data to avoid modifying the original dataset.   
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
    def explore_data(self):
        
        """
        Explores the dataset and returns some basic information about it.
        """
        print("Data Shape:", self.data.shape)
        print("\nData Types:\n", self.data.dtypes)
        print("\nMissing Values:\n", self.data.isnull().sum())
        print("\nData Description:\n", self.data.describe())
    
    # EXPLORE the dataset using sqlite extension and return some basic information about it.
    def explore_data_sqlite(self):
        
        """
        Explores the dataset using sqlite extension and returns some basic information about it.
        """

        # Create the connection to the in-memory SQLite database and store the data in a table called 'data'
        conn = sqlite3.connect(':memory:')
        # Store the data in a table called 'data'
        self.data.to_sql('data', conn, index=False)
        # Make a query to select the first 5 rows of the data table and print the result
        query = "SELECT * FROM data LIMIT 5"
        # Execute the query and store the result in a pandas DataFrame
        result = pd.read_sql_query(query, conn)
        # Display the result, but not printing, but using the display function from IPython.display to show the DataFrame in a more readable format
        from IPython.display import display
        display(result)
        # Close the connection to the SQLite database
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
    def column_names(self):
        
        """
        Returns the column names of the dataset.
        """
        return self.data.columns.tolist()
    
    # CLEAR the specified columns from the dataset =====> ["Unnamed: 0"]
    # this is not required for the preprocessing but it can be useful to remove unnecessary columns.
    def clear_columns(self, columns_to_clear: List[str]):
        
        """
        Clears the specified columns from the dataset.
        
        Parameters:
        columns_to_clear (list): List of column names to be cleared.
        """
        self.data.drop(columns=columns_to_clear, inplace=True)

    # Handle some column types transformations for columns
    # But this is a particular case of our dataset, so thez are optional so variables/columns can be different for other cases
    def transform_columns_types(self, transformation_type: str, columns: List[str], new_column_names: List[str]):
        
        """
        The function transforms a set of columns depending on the transformation type
        
        transformation types:
        - String_to_Numeric_specif: FOR OUR SPECIFIC DATASET, for columns with string values that can be converted to numeric values, but with specific cases to handle
        - String_to_Numeric: for columns with string values that can be converted to numeric values
        - String_to_Datetime: for columns with string values that can be converted to datetime values
        - Categorical_to_Numeric: for columns with categorical values that can be converted to numeric values
        """
        
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

        # Here second case
        elif transformation_type == "String_to_Numeric":
            for i, column in enumerate(columns):
                self.data[new_column_names[i]] = pd.to_numeric(self.data[column], errors='coerce')

        # Here the thirs case =====> Column: earliest_cr_line
        elif transformation_type == "String_to_Datetime":
            for i, column in enumerate(columns):
                self.data[new_column_names[i]] = pd.to_datetime(self.data[column], format='%b-%y', errors='coerce')

        # Here the fourth case
        elif transformation_type == "Categorical_to_Numeric":
            for i, column in enumerate(columns):
                self.data[new_column_names[i]] = self.data[column].astype('category').cat.codes
        
        # We get the defined reference date from the user input, and convert it to a datetime object
        @staticmethod
        def get_defined_reference_date(define_reference_date: str):
             defined_reference_date = pd.to_datetime(define_reference_date)
             return defined_reference_date
         
        # Specific function to get the days and months since the earliest credit line, based on the defined reference date
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
            print("\nDays since earliest credit line:\n", self.data['days_since_earliest_credit_line'].describe())
            print("\nMonths since earliest credit line:\n", self.data['months_since_earliest_credit_line'].describe())
        
        


        

        
    